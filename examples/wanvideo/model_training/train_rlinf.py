import torch, os, json
import numpy as np
from PIL import Image
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.utils import RLinfNpyDataset, LiberoRGBDDataset
from diffsynth.models.wan_video_dit import CrossBranchAttention
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Patch Start: 允许加载包含 set 的权重文件 ---
try:
    torch.serialization.add_safe_globals(['set', 'OrderedDict', 'builtins.set'])
except AttributeError:
    pass
# --- Patch End ---



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        context_noise_sigma=0.0,
        static_video_prob=0.0,
        use_wow_checkpoint=False,
        use_dual_branch=False,         # new: enable RGB/depth dual-branch
        depth_pretrain_checkpoint=None, # new: optional separate pretrain for depth branch
        rgb_pretrain_checkpoint=None,   # new: load rgb-only ckpt into pipe.dit then freeze it
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)

        # --- Patch Start: 加载 WoW 权重 ---
        if use_wow_checkpoint:
            wow_ckpt_path = "/opt/zsq/wow-world-model/dit_models/checkpoints/WoW-1-Wan-14B-600k/WoW_video_dit.pt"
            if os.path.exists(wow_ckpt_path):
                print(f"Overwriting with WoW checkpoint: {wow_ckpt_path}")
                state_dict = torch.load(wow_ckpt_path, map_location="cpu", weights_only=False)
                msg = self.pipe.dit.load_state_dict(state_dict, strict=False)
                print(f"WoW checkpoint loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            else:
                print(f"WoW checkpoint not found at {wow_ckpt_path}, skipping.")
        else:
            print("Not using WoW checkpoint.")
        # --- Patch End ---

        # --- RGB pretrain checkpoint (dual-branch freeze mode) ---
        if rgb_pretrain_checkpoint is not None and os.path.exists(rgb_pretrain_checkpoint):
            print(f"Loading RGB pretrain checkpoint into pipe.dit: {rgb_pretrain_checkpoint}")
            from diffsynth.models import load_state_dict as _load_sd
            sd = _load_sd(rgb_pretrain_checkpoint, torch_dtype=torch.bfloat16, device="cpu")
            # strip common prefix if present
            if all(k.startswith("pipe.dit.") for k in list(sd.keys())[:5]):
                sd = {k[len("pipe.dit."):]: v for k, v in sd.items()}
            msg = self.pipe.dit.load_state_dict(sd, strict=False)
            print(f"   Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        # ---

        # --- Dual-branch setup ---
        self.use_dual_branch = use_dual_branch
        if use_dual_branch:
            import copy
            print("🔀 Dual-branch mode: initializing depth_dit from RGB dit weights.")
            # Deep-copy the RGB DiT as the depth branch starting point
            self.pipe.depth_dit = copy.deepcopy(self.pipe.dit)
            if depth_pretrain_checkpoint is not None and os.path.exists(depth_pretrain_checkpoint):
                print(f"   Loading depth branch pretrain from: {depth_pretrain_checkpoint}")
                sd = torch.load(depth_pretrain_checkpoint, map_location="cpu", weights_only=False)
                msg = self.pipe.depth_dit.load_state_dict(sd, strict=False)
                print(f"   Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            # Build one CrossBranchAttention per DiT block (zero-init output proj)
            num_blocks = len(self.pipe.dit.blocks)
            dim = self.pipe.dit.dim
            num_heads = self.pipe.dit.blocks[0].num_heads
            self.pipe.cross_branch_attentions = torch.nn.ModuleList([
                CrossBranchAttention(dim=dim, num_heads=num_heads)
                for _ in range(num_blocks)
            ])
            print(f"   CrossBranchAttention modules: {num_blocks} (dim={dim}, heads={num_heads})")
        # ---

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # If dual-branch, also set depth_dit and cross_branch_attentions to train;
        # freeze pipe.dit (RGB branch) to save memory — it acts as a frozen encoder.
        if use_dual_branch:
            self.pipe.depth_dit.train()
            self.pipe.cross_branch_attentions.train()
            for p in self.pipe.depth_dit.parameters():
                p.requires_grad_(True)
            for p in self.pipe.cross_branch_attentions.parameters():
                p.requires_grad_(True)
            # Freeze RGB dit
            self.pipe.dit.eval()
            for p in self.pipe.dit.parameters():
                p.requires_grad_(False)
            print("   RGB dit frozen (requires_grad=False).")

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.context_noise_sigma = context_noise_sigma
        self.static_video_prob = static_video_prob


    def forward_preprocess(self, data):
        # === 静态样本增强 ===
        if self.training and self.static_video_prob > 0 and np.random.rand() < self.static_video_prob:
            first_frame = data["video"][0]
            data["video"] = [first_frame] * len(data["video"])
            if "action" in data:
                data["action"] = torch.zeros_like(data["action"])
                data["action"][:,-1] = -1
        # ============================
        inputs_posi = {}
        inputs_nega = {}

        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "idx":data["idx"] if "idx" in data else None,
        }

        if "action" in data:
            assert torch.isfinite(data["action"]).all(), f"Action tensor has NaN/Inf. Shape: {data['action'].shape}, min: {data['action'].min()}, max: {data['action'].max()}"
            inputs_shared["action"] = data["action"]

        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                if self.context_noise_sigma > 0:
                    img_arr = np.array(data["video"][0]).astype(np.float32)
                    noise = np.random.normal(0, self.context_noise_sigma, img_arr.shape)
                    img_noisy = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
                    inputs_shared["input_image"] = Image.fromarray(img_noisy)
                else:
                    inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        processed = {**inputs_shared, **inputs_posi}

        # --- Dual-branch: encode depth video and create depth noise ---
        if self.use_dual_branch and "depth" in data:
            depth_tensor = data["depth"].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            # depth_tensor: (T, 1, H, W) → expand to 3-ch and convert to PIL list for VAE encode
            # Actually encode directly via VAE as a grayscale-to-latent pass.
            # Replicate channel dim: (T, 1, H, W) -> (T, 3, H, W) so VAE can accept it.
            depth_3ch = depth_tensor.expand(-1, 3, -1, -1)  # (T, 3, H, W)
            # Convert to list of PIL Images for pipe.preprocess_video
            depth_pil = []
            for t in range(depth_3ch.shape[0]):
                arr = (depth_3ch[t].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                depth_pil.append(Image.fromarray(arr))
            self.pipe.load_models_to_device(["vae"])
            depth_video_tensor = self.pipe.preprocess_video(depth_pil).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            depth_input_latents = self.pipe.vae.encode(
                depth_video_tensor, device=self.pipe.device, tiled=False
            ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            depth_noise = torch.randn_like(depth_input_latents)
            processed["depth_input_latents"] = depth_input_latents
            processed["depth_noise"] = depth_noise

        return processed


    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--context_noise_sigma", type=float, default=10, help="Sigma of Gaussian noise added to the context frame")
    parser.add_argument("--static_video_prob", type=float, default=0.15, help="Probability of replacing the sample with a static video (action=0)")
    parser.add_argument("--use_wow_checkpoint", action="store_true",help="Whether to load the WoW checkpoint to overwrite the base model weights.")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval in epochs")
    parser.add_argument("--dataset", type=str, default="RLinfNpyDataset", help="Dataset type for training")
    # LiberoRGBDDataset arguments
    parser.add_argument("--libero_dataset_name", type=str, default=None, help="HuggingFace dataset name for LiberoRGBDDataset")
    parser.add_argument("--libero_context_frames", type=int, default=5, help="Number of context frames for LiberoRGBDDataset")
    parser.add_argument("--libero_prediction_frames", type=int, default=4, help="Number of prediction frames for LiberoRGBDDataset")
    parser.add_argument("--libero_image_size", type=int, nargs=2, default=[256, 256], help="Image size (H W) for LiberoRGBDDataset")
    parser.add_argument("--libero_action_dim", type=int, default=7, help="Action dimension for LiberoRGBDDataset")
    parser.add_argument("--libero_max_depth", type=float, default=5.0, help="Max depth value for LiberoRGBDDataset")
    # Dual-branch arguments
    parser.add_argument("--use_dual_branch", action="store_true", help="Enable dual-branch RGB+depth DiT with cross-attention exchange.")
    parser.add_argument("--depth_pretrain_checkpoint", type=str, default=None, help="Path to pretrained depth-only DiT checkpoint for depth branch init.")
    parser.add_argument("--rgb_pretrain_checkpoint", type=str, default=None, help="Path to rgb-only pretrained checkpoint; loaded into pipe.dit then frozen.")
    args = parser.parse_args()

    if args.dataset == "RLinfNpyDataset":
        dataset = RLinfNpyDataset(
            base_path=os.path.join(args.dataset_base_path, 'train_data'),
            repeat=args.dataset_repeat,
            num_frames=args.num_frames,
        )
        val_dataset = RLinfNpyDataset(
            base_path=os.path.join(args.dataset_base_path, 'val_data'),
            repeat=1,
            num_frames=args.num_frames
        )
    else:
        _libero_kwargs = dict(
            dataset_name=args.libero_dataset_name,
            context_frames=args.libero_context_frames,
            prediction_frames=args.libero_prediction_frames,
            image_size=tuple(args.libero_image_size),
            action_dim=args.libero_action_dim,
            max_depth=args.libero_max_depth,
        )
        val_dataset = LiberoRGBDDataset(**_libero_kwargs, split="validation", repeat=1)
        dataset = LiberoRGBDDataset(**_libero_kwargs, split="train", repeat=args.dataset_repeat)
        if val_dataset._fallback_split is not None:
            n_all = len(dataset.episodes)
            val_ratio = dataset._fallback_val_ratio
            train_end = int(n_all * (1.0 - val_ratio))
            dataset.episodes = dataset.episodes[:train_end]
            print(f"[LiberoRGBDDataset] Auto-split (fallback): "
                  f"using episodes [0, {train_end}) ({len(dataset.episodes)} episodes) as 'train'.")
    # ----------------------
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        context_noise_sigma=args.context_noise_sigma,
        static_video_prob=args.static_video_prob,
        use_wow_checkpoint=args.use_wow_checkpoint,
        use_dual_branch=args.use_dual_branch,
        depth_pretrain_checkpoint=args.depth_pretrain_checkpoint,
        rgb_pretrain_checkpoint=args.rgb_pretrain_checkpoint,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, val_dataset, model, model_logger, args=args)
