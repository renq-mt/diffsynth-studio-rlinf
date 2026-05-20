import imageio, os, torch, warnings, torchvision, argparse, json
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat

class RLinfNpyDataset(torch.utils.data.Dataset):
    """
    dataset_base_path/  
        rgb.npy  : [T, N, 3, H, W]
        traj.npy : [T, N, 3, H, W] 或 [T, N, ...]，需保证每帧能转成图片
    """

    def __init__(self, base_path='/opt/zsq/rlinf_dataset_1114_split/train_data', num_frames=9, repeat=1):
        self.base_path = base_path
        self.num_frames = num_frames
        self.repeat = repeat
        self.load_from_cache = False

        self.data_paths = []
        for step_name in sorted(os.listdir(base_path)):
            # step_path = os.path.join(base_path, step_name, "video/eval")
            step_path = os.path.join(base_path, step_name)
            if not os.path.isdir(step_path):
                continue
            for seed_name in sorted(os.listdir(step_path)):
                seed_path = os.path.join(step_path, seed_name)
                if os.path.isdir(seed_path):
                    self.data_paths.append(seed_path)
        if len(self.data_paths) == 0:
            raise ValueError(f"No valid data paths found under {base_path}")
        print(f'Found {len(self.data_paths)} data paths under {base_path}.')

        self.T_list = []
        self.N_list = []

        for p in self.data_paths:
            rgb_shape = np.load(os.path.join(p, "rgb.npy"), mmap_mode='r').shape
            T, N = rgb_shape[0], rgb_shape[1]
            self.T_list.append(T)
            self.N_list.append(N)

        self.total_env = sum(self.N_list)
        self.length = self.total_env * repeat 

        self.cum_N = np.cumsum(self.N_list)


    def __len__(self):
        return self.length
    def _locate_env(self, global_env_id):
        path_idx = np.searchsorted(self.cum_N, global_env_id, side='right')
        if path_idx == 0:
            env_id = global_env_id
        else:
            env_id = global_env_id - self.cum_N[path_idx - 1]
        return path_idx, env_id
    def __getitem__(self, idx):
        global_env_id = idx % self.total_env

        path_idx, env_id = self._locate_env(global_env_id)
        data_path = self.data_paths[path_idx]
        T = self.T_list[path_idx]
        rgb = np.load(os.path.join(data_path, "rgb.npy"), mmap_mode='r')
        actions = np.load(os.path.join(data_path, "actions.npy"), mmap_mode='r')


        if T > (self.num_frames-1):
            if np.random.rand() < 0.95:
                # 95% 随机采样
                if T >256:
                    start_idx = np.random.randint(0, 250)
                else:
                    start_idx = np.random.randint(0, T - self.num_frames + 2)

                consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)
                frame_ids = np.concatenate([[0], consecutive_ids])
            else:
                # 5% 使用硬编码 frame_ids
                frame_ids = np.array([0,0,0,0,0,1,2,3,4,5,6,7,8])
        else:
            raise ValueError(f"T={T} is too small for num_frames={self.num_frames}")
        

        video_np = rgb[frame_ids, env_id]  # shape [num_frames, 3, H, W]
        video_list = []
        for frame in video_np:
            if "/opt/zsq/rlinf_dataset_1114_split" in data_path:
                img = np.transpose(frame, (1, 2, 0))  # CHW → HWC
            else:
                img = frame
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255)
            video_list.append(Image.fromarray(img.astype(np.uint8)))

        # Action → Tensor
        action_np = actions[frame_ids, env_id]  # [num_frames, action_dim]
        action_tensor = torch.from_numpy(action_np).float()
        action_tensor[0] = torch.tensor([0., 0., 0., 0., 0., 0., -1.],
                                  dtype=action_tensor.dtype,
                                  device=action_tensor.device)
        return {
            "video": video_list,
            "reference_image": [video_list[0]],
            "action": action_tensor,
        }

class LiberoRGBDDataset(torch.utils.data.Dataset):
    """
    LIBERO RGB-D dataset loader for depth-conditioned world model training.

    Loads from HuggingFace LeRobot format datasets (e.g. binhng/libero_object_lerobot_mask_depth).
    Each sample contains context + prediction frames with RGB, depth, and actions.

    Dataset format:
        observation.images.image       : RGB frames (PIL Image)
        observation.images.image_depth : depth maps (PIL Image or ndarray)
        action                         : robot action vector (7D)
        episode_index                  : episode identifier
    """

    RGB_KEY = "observation.images.image"
    DEPTH_KEY = "observation.images.image_depth"
    ACTION_KEY = "action"
    EPISODE_KEY = "episode_index"

    def __init__(
        self,
        dataset_name: str,
        context_frames: int = 5,
        prediction_frames: int = 4,
        image_size=(256, 256),
        action_dim: int = 7,
        split: str = "train",
        repeat: int = 1,
    ):
        self.dataset_name = dataset_name
        from datasets import load_dataset
        self.context_frames = context_frames
        self.prediction_frames = prediction_frames
        self.window_size = context_frames + prediction_frames
        self.num_frames = context_frames + prediction_frames  # For consistency with RLinfNpyDataset
        self.image_size = image_size
        self.action_dim = action_dim

        self.repeat = repeat

        self._fallback_val_ratio = 0.1   # ratio reserved for auto val split
        self._fallback_split = None       # non-None when we fell back to "train"
        try:
            self.dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        except ValueError as e:
            if split != "train":
                warnings.warn(
                    f"Split '{split}' not found in dataset '{dataset_name}' ({e}). "
                    f"Falling back to split='train' and using the last "
                    f"{int(self._fallback_val_ratio * 100)}% of episodes as '{split}'."
                )
                self.dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
                self._fallback_split = split
            else:
                raise
        self._build_index()
        # Auto-split: when there is no dedicated val/test split in the dataset,
        # carve out the last 10% of episodes for validation and keep the
        # first 90% for training.
        if self._fallback_split is not None:
            n = len(self.episodes)
            val_start = int(n * (1.0 - self._fallback_val_ratio))
            # requested split was validation/test → keep tail
            self.episodes = self.episodes[val_start:]
            print(f"[LiberoRGBDDataset] Auto-split (fallback): "
                  f"using episodes [{val_start}, {n}) ({len(self.episodes)} episodes) "
                  f"as '{self._fallback_split}'.")

    def _build_index(self):
        if self._build_index_from_meta():
            return
        self.episodes = []
        episode_start = 0
        current_ep = None
        for i in range(len(self.dataset)):
            ep = self.dataset[i].get(self.EPISODE_KEY, i)
            if current_ep is not None and ep != current_ep:
                self._add_episode(episode_start, i)
                episode_start = i
            current_ep = ep
        if current_ep is not None:
            self._add_episode(episode_start, len(self.dataset))

    def _build_index_from_meta(self):
        if not os.path.isdir(self.dataset_name):
            return False
        meta_path = os.path.join(self.dataset_name, "meta", "episodes.jsonl")
        if not os.path.exists(meta_path):
            return False

        self.episodes = []
        episode_start = 0
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                episode = json.loads(line)
                episode_len = int(episode["length"])
                episode_end = episode_start + episode_len
                self._add_episode(episode_start, episode_end)
                episode_start = episode_end
        return True

    def _add_episode(self, start, end):
        """Add episode information for random sampling."""
        if end - start < self.window_size:
            return  # Skip episodes that are too short
        # Store episode info: (start_index, end_index, total_length)
        self.episodes.append((start, end, end - start))

    def _load_rgb(self, item):
        """Load RGB image as PIL Image (uint8 [0, 255])."""
        import numpy as np
        img = item.get(self.RGB_KEY)
        if img is None:
            # Return black image
            return Image.new("RGB", (self.image_size[1], self.image_size[0]), (0, 0, 0))

        if not isinstance(img, Image.Image):
            # Convert array to PIL Image
            img_array = np.array(img)
            # If image is in [0, 1] float range, convert to [0, 255] uint8
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

        # Resize to target size
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        return img

    def _load_depth(self, item):
        """Load depth map as PIL Image (uint8 [0, 255]) with RGB channels."""
        import numpy as np
        depth_obj = item.get(self.DEPTH_KEY)
        if depth_obj is None:
            # Return white depth map (max depth) with RGB channels.
            return Image.new("RGB", (self.image_size[1], self.image_size[0]), (255, 255, 255))

        depth_arr = np.array(depth_obj)
        depth_dtype = depth_arr.dtype

        if depth_arr.ndim == 3:
            if depth_arr.shape[2] == 1:
                depth_arr = np.repeat(depth_arr, 3, axis=2)
            elif depth_arr.shape[2] > 3:
                depth_arr = depth_arr[:, :, :3]

            depth_rgb = depth_arr.astype(np.float32)
            if np.issubdtype(depth_dtype, np.integer):
                depth_rgb = depth_rgb.clip(0, 255).astype(np.uint8)
            elif depth_rgb.max() <= 1.0 and depth_rgb.max() > 0:
                depth_rgb = (depth_rgb * 255).clip(0, 255).astype(np.uint8)
            else:
                depth_rgb = depth_rgb.clip(0, 255).astype(np.uint8)

            return Image.fromarray(depth_rgb, mode="RGB").resize(
                (self.image_size[1], self.image_size[0]), Image.NEAREST
            )

        depth = depth_arr.astype(np.float32)
        assert depth.ndim == 2, f"Depth should be 2D after processing, got shape {depth.shape}"

        if np.issubdtype(depth_dtype, np.integer):
            depth_uint8 = depth.clip(0, 255).astype(np.uint8)
        elif depth.max() <= 1.0 and depth.max() > 0:
            depth_uint8 = (depth * 255).clip(0, 255).astype(np.uint8)
        else:
            depth_uint8 = depth.clip(0, 255).astype(np.uint8)
        depth_rgb = np.repeat(depth_uint8[:, :, None], 3, axis=2)
        return Image.fromarray(depth_rgb, mode="RGB").resize(
            (self.image_size[1], self.image_size[0]), Image.NEAREST
        )

    def _load_action(self, item) -> torch.Tensor:
        import numpy as np
        action = item.get(self.ACTION_KEY)
        if action is None:
            return torch.zeros(self.action_dim)
        if isinstance(action, np.ndarray):
            return torch.from_numpy(action.astype(np.float32))
        return torch.tensor(action, dtype=torch.float32)

    def __len__(self):
        return len(self.episodes) * self.repeat

    def __getitem__(self, idx):
        """Random sampling strategy consistent with RLinfNpyDataset."""
        episode_idx = idx % len(self.episodes)
        episode_start, episode_end, episode_length = self.episodes[episode_idx]

        # Debug: print first few calls to understand data
        if idx < 3:
            print(f"[DEBUG] idx={idx}, episode_idx={episode_idx}, episode_length={episode_length}, num_frames={self.num_frames}")

        # Random sampling strategy (consistent with RLinfNpyDataset)
        import numpy as np

        if episode_length > (self.num_frames - 1):
            if np.random.rand() < 0.95:
                # 95% 随机采样 - 第0帧固定 + 从start_idx开始的连续片段（与RLinfNpyDataset一致）
                if episode_length > 256:
                    start_idx = np.random.randint(0, 250)
                else:
                    start_idx = np.random.randint(0, episode_length - self.num_frames + 2)

                consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)
                frame_ids = np.concatenate([[0], consecutive_ids])
                # Ensure frame_ids don't exceed episode length
                frame_ids = np.clip(frame_ids, 0, episode_length - 1)
            else:
                # 5% 使用硬编码 frame_ids pattern - 第0帧重复5次，然后连续帧
                frame_ids = np.array([0,0,0,0,0,1,2,3,4,5,6,7,8][:self.num_frames])
                # Ensure we have enough frames
                while len(frame_ids) < self.num_frames:
                    frame_ids = np.append(frame_ids, frame_ids[-1] + 1)
                    frame_ids[-1] = min(frame_ids[-1], episode_length - 1)
        else:
            raise ValueError(f"Episode length={episode_length} is too small for num_frames={self.num_frames}")

        # Load frames using the sampled frame_ids
        rgb_frames, depth_frames, actions = [], [], []
        for frame_id in frame_ids:
            actual_idx = int(episode_start + frame_id)
            item = self.dataset[actual_idx]
            rgb_frames.append(self._load_rgb(item))
            depth_frames.append(self._load_depth(item))
            actions.append(self._load_action(item))

        # rgb_frames are already PIL Images (uint8 [0, 255])
        video_list = rgb_frames

        action_tensor = torch.stack(actions)  # (num_frames, action_dim)
        # Zero out first frame action (context frame convention)
        action_tensor[0] = torch.tensor([0., 0., 0., 0., 0., 0., -1.])

        # Keep depth aligned with video semantics: return RGB PIL frames directly.
        assert torch.isfinite(action_tensor).all(), f"Action has NaN/Inf: {action_tensor}"
        return {
            "video": video_list,
            "reference_image": [video_list[0]],
            "action": action_tensor,
            "depth": depth_frames,       # list of T RGB PIL Images
        }

    def load_episode_sequential(self, episode_idx, length=None):
        """Return ``(rgb_list, actions_np, depth_pil_list)`` for the first
        ``length`` consecutive frames of ``episodes[episode_idx]``.

        Used by inference / validation to bypass __getitem__'s random sampling.
        """
        ep_start, ep_end, ep_len = self.episodes[episode_idx]
        n = ep_end - ep_start if length is None else min(int(length), ep_end - ep_start)
        rgb_list, depth_list, action_list = [], [], []
        for i in range(ep_start, ep_start + n):
            item = self.dataset[i]
            rgb_list.append(self._load_rgb(item))
            depth_list.append(self._load_depth(item))
            action_list.append(self._load_action(item).numpy())
        return rgb_list, np.stack(action_list, axis=0), depth_list


class RLinfLerobotDataset(torch.utils.data.Dataset):
    """
    RLinf rollout LeRobot v2.x dataset loader (with multi-root aggregation).

    This loader is intentionally separate from ``LiberoRGBDDataset`` because
    the rlinf rollout output uses a different schema and ships raw local
    LeRobot directories (no HuggingFace hub).

    ``dataset_name`` accepts three forms:
      1. A single LeRobot root containing ``meta/info.json``.
      2. A parent directory; the loader will scan up to two levels deep for
         ``meta/info.json`` files and aggregate every match (e.g.
         ``rollout_lerobot/rank_*/id_*``).
      3. A list/tuple of paths, each of which is itself form (1) or (2).

    Each LeRobot root has its own ``episode_index`` namespace starting at 0,
    so episodes from different roots are tagged with a root index internally to
    avoid collisions.

    Expected per-root layout::

        <root>/
          meta/{info.json, episodes.jsonl, tasks.jsonl}
          data/chunk-XXX/episode_XXXXXX.parquet
          images/                 # optional PNG mirror; not required

    Parquet schema (image columns are ``struct<bytes, path>`` with the bytes
    embedded inline)::

        image                            : main camera RGB (PNG bytes)
        observation.images.image_depth   : main camera depth (PNG bytes)
        actions                          : 7-D robot action vector

    Output dict matches :class:`LiberoRGBDDataset`::

        {"video": [PIL]*T, "reference_image": [PIL], "action": Tensor(T,7),
         "depth": [PIL]*T}
    """

    RGB_KEY = "image"
    DEPTH_KEY = "observation.images.image_depth"
    ACTION_KEY = "actions"
    EPISODE_KEY = "episode_index"

    # How deep to scan when ``dataset_name`` is a parent directory.
    _AGGREGATE_MAX_DEPTH = 2

    def __init__(
        self,
        dataset_name,
        context_frames: int = 5,
        prediction_frames: int = 4,
        image_size=(256, 256),
        action_dim: int = 7,
        split: str = "train",
        repeat: int = 1,
        parquet_cache_size: int = 4,
    ):
        import pyarrow.parquet as pq  # fail fast on missing dep

        self.dataset_name = dataset_name
        self.context_frames = context_frames
        self.prediction_frames = prediction_frames
        self.window_size = context_frames + prediction_frames
        self.num_frames = self.window_size  # consistency with the other datasets
        self.image_size = image_size
        self.action_dim = action_dim
        self.repeat = repeat

        self._fallback_val_ratio = 0.1
        self._fallback_split = None

        # 1) Resolve one or more LeRobot roots.
        roots = self._resolve_roots(dataset_name)
        if not roots:
            raise FileNotFoundError(
                f"[RLinfLerobotDataset] no LeRobot root with meta/info.json found "
                f"under {dataset_name!r} (searched up to depth {self._AGGREGATE_MAX_DEPTH})"
            )
        self._roots = roots                       # List[str]
        self._root_info = []                      # List[dict]: per-root info

        # 2) Load per-root metadata.
        for root in self._roots:
            with open(os.path.join(root, "meta", "info.json"), "r", encoding="utf-8") as f:
                info = json.load(f)
            self._root_info.append({
                "data_path_template": info.get(
                    "data_path",
                    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                ),
                "chunks_size": int(info.get("chunks_size", 1000)),
            })

        # 3) Schema check against the first available parquet of the first root.
        first_path = None
        for ri, root in enumerate(self._roots):
            for rec in self._iter_episode_records(ri):
                path = self._episode_parquet_path(ri, int(rec["episode_index"]))
                if os.path.isfile(path):
                    first_path = path
                    break
            if first_path is not None:
                break
        if first_path is None:
            raise FileNotFoundError(
                f"[RLinfLerobotDataset] no parquet file found under any of {self._roots}"
            )
        schema = pq.read_schema(first_path)
        cols = set(schema.names)
        missing = [k for k in (self.RGB_KEY, self.DEPTH_KEY, self.ACTION_KEY) if k not in cols]
        if missing:
            raise KeyError(
                f"[RLinfLerobotDataset] required columns missing in {first_path}: {missing}. "
                f"Available columns: {sorted(cols)}"
            )

        # 4) Build the aggregated episode list.
        # self.episodes: List[(root_idx, episode_idx_in_root, length)]
        # self._episode_paths: (root_idx, episode_idx_in_root) -> parquet path
        self.episodes = []
        self._episode_paths = {}
        per_root_kept = [0] * len(self._roots)
        per_root_total = [0] * len(self._roots)
        for ri, root in enumerate(self._roots):
            for rec in self._iter_episode_records(ri):
                ep_idx = int(rec["episode_index"])
                ep_len = int(rec["length"])
                per_root_total[ri] += 1
                path = self._episode_parquet_path(ri, ep_idx)
                if not os.path.isfile(path):
                    warnings.warn(
                        f"[RLinfLerobotDataset] missing parquet for root={root} "
                        f"episode={ep_idx}: {path}"
                    )
                    continue
                self._episode_paths[(ri, ep_idx)] = path
                if ep_len >= self.window_size:
                    self.episodes.append((ri, ep_idx, ep_len))
                    per_root_kept[ri] += 1

        # 5) Validation split fallback (rlinf rollout dumps have no dedicated
        # split): carve out the last 10% of the aggregated episode list when
        # caller asks for non-"train".
        if split != "train":
            self._fallback_split = split
            n = len(self.episodes)
            val_start = int(n * (1.0 - self._fallback_val_ratio))
            self.episodes = self.episodes[val_start:]
            print(f"[RLinfLerobotDataset] Auto-split (fallback): "
                  f"using episodes [{val_start}, {n}) ({len(self.episodes)} episodes) "
                  f"as '{split}'.")

        # 6) Per-root LRU cache. Caching is shared across roots; the key is
        # (root_idx, episode_idx_in_root).
        self._parquet_cache = {}
        self._parquet_cache_order = []
        self._parquet_cache_max = max(1, int(parquet_cache_size))

        # Summary log.
        if len(self._roots) == 1:
            print(f"[RLinfLerobotDataset] root='{self._roots[0]}' "
                  f"episodes={len(self.episodes)} split='{split}' "
                  f"keys: rgb={self.RGB_KEY!r} depth={self.DEPTH_KEY!r} "
                  f"action={self.ACTION_KEY!r}")
        else:
            print(f"[RLinfLerobotDataset] aggregated {len(self._roots)} roots "
                  f"under {dataset_name!r}, total_episodes={len(self.episodes)} "
                  f"split='{split}' keys: rgb={self.RGB_KEY!r} "
                  f"depth={self.DEPTH_KEY!r} action={self.ACTION_KEY!r}")
            for ri, root in enumerate(self._roots):
                print(f"  - [{ri}] {root}  "
                      f"kept={per_root_kept[ri]}/{per_root_total[ri]}")

    # -------- root discovery ----------------------------------------------

    @staticmethod
    def _is_lerobot_root(p):
        return (
            os.path.isdir(p)
            and os.path.isfile(os.path.join(p, "meta", "info.json"))
            and os.path.isfile(os.path.join(p, "meta", "episodes.jsonl"))
            and os.path.isdir(os.path.join(p, "data"))
        )

    @classmethod
    def _resolve_roots(cls, dataset_name):
        """Expand ``dataset_name`` (str, parent dir, or list) into a sorted list
        of valid single LeRobot roots."""
        import glob
        if isinstance(dataset_name, (list, tuple)):
            raw_paths = [str(p) for p in dataset_name]
        else:
            raw_paths = [str(dataset_name)]

        resolved = []
        seen = set()
        for raw in raw_paths:
            if not os.path.isdir(raw):
                warnings.warn(f"[RLinfLerobotDataset] path is not a directory: {raw}")
                continue
            if cls._is_lerobot_root(raw):
                if raw not in seen:
                    resolved.append(raw)
                    seen.add(raw)
                continue
            # Otherwise, search downward for any direct meta/info.json.
            found_here = []
            for depth in range(1, cls._AGGREGATE_MAX_DEPTH + 1):
                pattern = os.path.join(raw, *(["*"] * depth), "meta", "info.json")
                for info_path in sorted(glob.glob(pattern)):
                    root = os.path.dirname(os.path.dirname(info_path))
                    if cls._is_lerobot_root(root) and root not in seen:
                        found_here.append(root)
                        seen.add(root)
                if found_here:
                    # Stop at the shallowest depth that produced any hit so
                    # we don't double-count nested layouts.
                    break
            if not found_here:
                warnings.warn(
                    f"[RLinfLerobotDataset] no LeRobot root found under {raw} "
                    f"(searched up to depth {cls._AGGREGATE_MAX_DEPTH})"
                )
            resolved.extend(found_here)

        return resolved

    def _iter_episode_records(self, root_idx):
        path = os.path.join(self._roots[root_idx], "meta", "episodes.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    # -------- parquet IO ---------------------------------------------------

    def _episode_parquet_path(self, root_idx, ep_idx):
        info = self._root_info[root_idx]
        chunk = ep_idx // info["chunks_size"]
        rel = info["data_path_template"].format(
            episode_chunk=chunk, episode_index=ep_idx,
        )
        return os.path.join(self._roots[root_idx], rel)

    def _load_episode_table(self, root_idx, ep_idx):
        cache_key = (root_idx, ep_idx)
        cached = self._parquet_cache.get(cache_key)
        if cached is not None:
            return cached
        import pyarrow.parquet as pq
        table = pq.read_table(self._episode_paths[cache_key])
        self._parquet_cache[cache_key] = table
        self._parquet_cache_order.append(cache_key)
        while len(self._parquet_cache_order) > self._parquet_cache_max:
            evict = self._parquet_cache_order.pop(0)
            self._parquet_cache.pop(evict, None)
        return table

    def _read_row(self, root_idx, ep_idx, frame_id):
        table = self._load_episode_table(root_idx, ep_idx)
        row = table.slice(int(frame_id), 1).to_pylist()[0]
        row[self.RGB_KEY] = self._materialize_image_cell(
            row.get(self.RGB_KEY), self.RGB_KEY, root_idx, ep_idx, frame_id,
        )
        row[self.DEPTH_KEY] = self._materialize_image_cell(
            row.get(self.DEPTH_KEY), self.DEPTH_KEY, root_idx, ep_idx, frame_id,
        )
        return row

    def _materialize_image_cell(self, cell, key, root_idx, ep_idx, frame_id):
        """Decode struct<bytes,path> dicts into PIL Images, falling back to
        the mirrored PNG files under ``<root>/images/<key>/episode_*/frame_*.png``."""
        import io
        if isinstance(cell, Image.Image):
            return cell
        rel_path = None
        if isinstance(cell, dict):
            data = cell.get("bytes")
            if data:
                try:
                    return Image.open(io.BytesIO(data))
                except Exception:
                    pass
            rel_path = cell.get("path")
        elif isinstance(cell, str):
            rel_path = cell

        root = self._roots[root_idx]
        images_root = os.path.join(root, "images", key)
        candidates = [
            os.path.join(
                images_root, f"episode_{ep_idx:06d}", f"frame_{int(frame_id):06d}.png"
            ),
        ]
        if rel_path:
            candidates.insert(0, os.path.join(images_root, f"episode_{ep_idx:06d}", rel_path))
            candidates.insert(0, os.path.join(images_root, rel_path))
        for p in candidates:
            if os.path.isfile(p):
                return Image.open(p)
        return cell  # let _load_rgb/_load_depth deal with the missing case

    # -------- field decoders (mirrors LiberoRGBDDataset semantics) --------

    def _load_rgb(self, img):
        import numpy as np
        if img is None:
            return Image.new("RGB", (self.image_size[1], self.image_size[0]), (0, 0, 0))
        if not isinstance(img, Image.Image):
            img_array = np.array(img)
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

    def _load_depth(self, depth_obj):
        import numpy as np
        if depth_obj is None:
            return Image.new("RGB", (self.image_size[1], self.image_size[0]), (255, 255, 255))

        depth_arr = np.array(depth_obj)
        depth_dtype = depth_arr.dtype

        if depth_arr.ndim == 3:
            if depth_arr.shape[2] == 1:
                depth_arr = np.repeat(depth_arr, 3, axis=2)
            elif depth_arr.shape[2] > 3:
                depth_arr = depth_arr[:, :, :3]

            depth_rgb = depth_arr.astype(np.float32)
            if np.issubdtype(depth_dtype, np.integer):
                depth_rgb = depth_rgb.clip(0, 255).astype(np.uint8)
            elif depth_rgb.max() <= 1.0 and depth_rgb.max() > 0:
                depth_rgb = (depth_rgb * 255).clip(0, 255).astype(np.uint8)
            else:
                depth_rgb = depth_rgb.clip(0, 255).astype(np.uint8)
            return Image.fromarray(depth_rgb, mode="RGB").resize(
                (self.image_size[1], self.image_size[0]), Image.NEAREST
            )

        depth = depth_arr.astype(np.float32)
        assert depth.ndim == 2, f"Depth should be 2D after processing, got shape {depth.shape}"
        if np.issubdtype(depth_dtype, np.integer):
            depth_uint8 = depth.clip(0, 255).astype(np.uint8)
        elif depth.max() <= 1.0 and depth.max() > 0:
            depth_uint8 = (depth * 255).clip(0, 255).astype(np.uint8)
        else:
            depth_uint8 = depth.clip(0, 255).astype(np.uint8)
        depth_rgb = np.repeat(depth_uint8[:, :, None], 3, axis=2)
        return Image.fromarray(depth_rgb, mode="RGB").resize(
            (self.image_size[1], self.image_size[0]), Image.NEAREST
        )

    def _load_action(self, action):
        import numpy as np
        if action is None:
            return torch.zeros(self.action_dim)
        if isinstance(action, np.ndarray):
            return torch.from_numpy(action.astype(np.float32))
        return torch.tensor(action, dtype=torch.float32)

    # -------- sampling -----------------------------------------------------

    def _sample_frame_ids(self, episode_length):
        """Mirrors LiberoRGBDDataset / RLinfNpyDataset sampling policy."""
        import numpy as np
        if episode_length <= self.num_frames - 1:
            raise ValueError(
                f"Episode length={episode_length} is too small for num_frames={self.num_frames}"
            )
        if np.random.rand() < 0.95:
            if episode_length > 256:
                start_idx = np.random.randint(0, 250)
            else:
                start_idx = np.random.randint(0, episode_length - self.num_frames + 2)
            consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)
            frame_ids = np.concatenate([[0], consecutive_ids])
            frame_ids = np.clip(frame_ids, 0, episode_length - 1)
        else:
            frame_ids = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8][: self.num_frames])
            while len(frame_ids) < self.num_frames:
                frame_ids = np.append(frame_ids, frame_ids[-1] + 1)
                frame_ids[-1] = min(frame_ids[-1], episode_length - 1)
        return frame_ids

    def __len__(self):
        return len(self.episodes) * self.repeat

    def __getitem__(self, idx):
        slot = idx % len(self.episodes)
        root_idx, ep_idx, episode_length = self.episodes[slot]

        if idx < 3:
            print(f"[DEBUG] RLinfLerobot idx={idx}, slot={slot}, "
                  f"root_idx={root_idx}, ep_idx={ep_idx}, "
                  f"episode_length={episode_length}, num_frames={self.num_frames}")

        frame_ids = self._sample_frame_ids(episode_length)

        rgb_frames, depth_frames, actions = [], [], []
        for frame_id in frame_ids:
            row = self._read_row(root_idx, ep_idx, int(frame_id))
            rgb_frames.append(self._load_rgb(row.get(self.RGB_KEY)))
            depth_frames.append(self._load_depth(row.get(self.DEPTH_KEY)))
            actions.append(self._load_action(row.get(self.ACTION_KEY)))

        action_tensor = torch.stack(actions)
        # Zero out first frame action (context frame convention, matches LiberoRGBDDataset)
        action_tensor[0] = torch.tensor([0., 0., 0., 0., 0., 0., -1.])
        assert torch.isfinite(action_tensor).all(), f"Action has NaN/Inf: {action_tensor}"
        return {
            "video": rgb_frames,
            "reference_image": [rgb_frames[0]],
            "action": action_tensor,
            "depth": depth_frames,
        }


class RLinfPickleDataset(torch.utils.data.Dataset):
    """
    Loader for RLinf ``CollectEpisode`` pickle dumps.

    Each pickle file is a single completed episode produced by
    ``rlinf.envs.wrappers.collect_episode.CollectEpisode`` when
    ``export_format="pickle"``. The directory typically contains files named
    ``rank_{R}_env_{E}_episode_{I}_step_{S}_{success|fail}.pkl`` (optionally
    ``.pkl.gz``).

    Per-episode pickle payload (relevant fields)::

        {
          "success": bool,
          "observations": list[T+1] of {
                "main_images":  Tensor(H,W,3) uint8,
                "main_depths":  Tensor(H,W,1) float32,
                "wrist_images": Tensor(H,W,3) uint8,
                "wrist_depths": Tensor(H,W,1) float32,
                "states":       Tensor(state_dim,) float32,
                "task_descriptions": str,
              },
          "actions": list[T] of ndarray(action_dim,) float (the leading
                     reset-obs entry is dropped here so len(actions)==T),
          ...
        }

    Output dict matches :class:`LiberoRGBDDataset` / :class:`RLinfLerobotDataset`::

        {"video": [PIL]*T, "reference_image": [PIL], "action": Tensor(T,7),
         "depth": [PIL]*T}

    Args:
        dataset_name: Either a directory containing ``*.pkl`` / ``*.pkl.gz``
            files, or a list/tuple of such directories.
        context_frames / prediction_frames: Sampling window =
            context_frames + prediction_frames frames per item.
        image_size: ``(H, W)`` target resize.
        action_dim: Action vector length. Only the leading ``action_dim`` dims
            of the recorded action are kept.
        split: ``"train"`` returns the leading 90% of episodes, anything else
            returns the trailing 10% (mirrors the other RLinf loaders).
        repeat: Dataset length multiplier for epoch sizing.
        episode_cache_size: LRU cache size for materialized episode payloads.
            Each entry holds the full episode (T frames of image + depth), so
            keep this small to bound memory.
        only_success: Drop episodes whose pickle filename ends in ``_fail`` /
            whose ``success`` field is False. Default False because rollout
            dumps may be entirely unsuccessful and world-model training does
            not require task success.
        rgb_key / depth_key / action_key: Override observation field names if
            the upstream wrapper saves them differently.
    """

    DEFAULT_RGB_KEYS = ("main_images", "image", "full_image")
    DEFAULT_DEPTH_KEYS = ("main_depths", "main_depth")

    _FNAME_RE = None  # lazy compile

    def __init__(
        self,
        dataset_name,
        context_frames: int = 5,
        prediction_frames: int = 4,
        image_size=(256, 256),
        action_dim: int = 7,
        split: str = "train",
        repeat: int = 1,
        episode_cache_size: int = 2,
        only_success: bool = False,
        rgb_key=None,
        depth_key=None,
        action_key: str = "actions",
    ):
        import glob, re

        self.dataset_name = dataset_name
        self.context_frames = context_frames
        self.prediction_frames = prediction_frames
        self.window_size = context_frames + prediction_frames
        self.num_frames = self.window_size
        self.image_size = image_size
        self.action_dim = action_dim
        self.repeat = repeat
        self.only_success = only_success
        self.action_key = action_key

        self._rgb_keys = (rgb_key,) if isinstance(rgb_key, str) else (
            tuple(rgb_key) if rgb_key else self.DEFAULT_RGB_KEYS
        )
        self._depth_keys = (depth_key,) if isinstance(depth_key, str) else (
            tuple(depth_key) if depth_key else self.DEFAULT_DEPTH_KEYS
        )

        self._fallback_val_ratio = 0.1
        self._fallback_split = None

        # Resolve directories.
        if isinstance(dataset_name, (list, tuple)):
            roots = [str(p) for p in dataset_name]
        else:
            roots = [str(dataset_name)]

        # Scan for episode files.
        files = []
        for root in roots:
            if not os.path.isdir(root):
                warnings.warn(f"[RLinfPickleDataset] not a directory: {root}")
                continue
            for pattern in ("*.pkl", "*.pkl.gz"):
                files.extend(sorted(glob.glob(os.path.join(root, pattern))))
        if not files:
            raise FileNotFoundError(
                f"[RLinfPickleDataset] no .pkl / .pkl.gz files found under {dataset_name!r}"
            )

        if RLinfPickleDataset._FNAME_RE is None:
            RLinfPickleDataset._FNAME_RE = re.compile(
                r"rank_(\d+)_env_(\d+)_episode_(\d+)_step_(\d+)_(success|fail)"
            )

        # episodes: list of dicts with metadata.
        # We CANNOT pickle.load every file at construction time — each RLinf
        # rollout pickle is ~470 MB (full 512-step LIBERO trajectory + RGB +
        # depth observations), so deserializing all 584 files takes >30 min
        # and stalls training launch.
        #
        # Instead, two-layer defense:
        #   1. Cheap pre-filter by file size: any pickle smaller than
        #      MIN_FILE_BYTES cannot possibly hold a full sampling window of
        #      (image+depth+state+action) per step. On 256x256 LIBERO rollouts,
        #      a real step is ~1 MB after wrapping all observation fields. A
        #      file <512 KB is one of the known "ghost success" stubs RLinf's
        #      CollectEpisode wrapper produces when the env stays terminated
        #      after task success (length=1, 80 such files seen in this dump).
        #   2. Lazy fallback in __getitem__: if a surviving pickle still
        #      turns out shorter than window_size, drop it from self.episodes
        #      and resample. This keeps construction O(files) without
        #      deserializing anything.
        # Strategy: drop any pickle smaller than MIN_FILE_BYTES. Real RLinf
        # rollouts on 256x256 LIBERO are ~470 MB (full 512-step trajectories).
        # The 80 known "ghost success" stubs (length=1) are ~0.9 MB each. So a
        # threshold of 2 MB cleanly separates them without bisecting any
        # genuine trajectory.
        MIN_FILE_BYTES = 2 * 1024 * 1024

        self.episodes = []
        n_size_pruned = 0
        n_dropped_filter = 0
        for f in files:
            base = os.path.basename(f)
            m = RLinfPickleDataset._FNAME_RE.search(base)
            success_hint = None
            if m is not None:
                success_hint = (m.group(5) == "success")
            if self.only_success and success_hint is False:
                n_dropped_filter += 1
                continue
            try:
                size = os.path.getsize(f)
            except OSError:
                size = -1
            if size != -1 and size < MIN_FILE_BYTES:
                n_size_pruned += 1
                continue
            self.episodes.append({
                "path": f,
                "success_hint": success_hint,
                "length": None,  # filled lazily on first __getitem__ load
            })

        if not self.episodes:
            raise RuntimeError(
                f"[RLinfPickleDataset] all {len(files)} files were filtered out "
                f"(only_success={self.only_success}, dropped={n_dropped_filter}). "
                f"Disable --rlinf_pkl_only_success or check the rollout dump."
            )

        # Sort by filename for a stable order across splits.
        self.episodes.sort(key=lambda e: e["path"])

        # Fallback train/val split (these rollouts have no dedicated split).
        if split != "train":
            self._fallback_split = split
            n = len(self.episodes)
            val_start = int(n * (1.0 - self._fallback_val_ratio))
            self.episodes = self.episodes[val_start:]
            print(f"[RLinfPickleDataset] Auto-split (fallback): "
                  f"using episodes [{val_start}, {n}) "
                  f"({len(self.episodes)} episodes) as '{split}'.")

        self._cache = {}
        self._cache_order = []
        self._cache_max = max(1, int(episode_cache_size))

        print(f"[RLinfPickleDataset] roots={roots} files_scanned={len(files)} "
              f"episodes={len(self.episodes)} split='{split}' "
              f"only_success={self.only_success} "
              f"window={self.window_size} action_dim={self.action_dim}")
        if n_dropped_filter:
            print(f"[RLinfPickleDataset]   dropped_by_filter={n_dropped_filter}")
        if n_size_pruned:
            print(f"[RLinfPickleDataset]   dropped_size_too_small(<{MIN_FILE_BYTES / 1024 / 1024:.1f}MB)={n_size_pruned}")

    # -------- pickle IO ----------------------------------------------------

    @staticmethod
    def _open_episode(path):
        """Load a pickle (auto-detects gzip via filename) and return the payload."""
        import gzip, pickle
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as fh:
            return pickle.load(fh)

    def _get_episode(self, slot):
        meta = self.episodes[slot]
        path = meta["path"]
        cached = self._cache.get(path)
        if cached is not None:
            return cached

        payload = self._open_episode(path)
        actions = payload.get(self.action_key, [])
        observations = payload.get("observations", [])
        # Align actions to observations: the wrapper prepends the reset obs, so
        # obs[0] has no preceding action and obs[t] corresponds to action[t-1].
        if len(observations) == len(actions) + 1:
            dummy_action = np.zeros(self.action_dim, dtype=np.float32)
            if self.action_dim >= 7:
                dummy_action[6] = -1.0
            observations = observations[:-1]
            actions = [dummy_action] + list(actions[:-1])
        elif len(observations) != len(actions):
            n = min(len(observations), len(actions))
            observations = observations[:n]
            actions = list(actions[:n])

        entry = {
            "observations": observations,
            "actions": actions,
            "success": bool(payload.get("success", meta.get("success_hint") or False)),
            "length": len(actions),
        }
        self._cache[path] = entry
        self._cache_order.append(path)
        while len(self._cache_order) > self._cache_max:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return entry

    # -------- field decoders (mirror LiberoRGBDDataset semantics) ----------

    def _to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def _pick_field(self, obs, keys):
        if not isinstance(obs, dict):
            return None
        for k in keys:
            if k in obs and obs[k] is not None:
                return obs[k]
        return None

    def _load_rgb(self, obs):
        arr = self._to_numpy(self._pick_field(obs, self._rgb_keys))
        if arr is None:
            return Image.new("RGB", (self.image_size[1], self.image_size[0]), (0, 0, 0))
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        return img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

    def _load_depth(self, obs):
        arr = self._to_numpy(self._pick_field(obs, self._depth_keys))
        if arr is None:
            return Image.new("RGB", (self.image_size[1], self.image_size[0]), (255, 255, 255))
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # Squeeze trailing singleton channel to 2D.
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # Normalize per-frame to [0, 255] just like RLinfLerobotDataset does for
        # depth images saved by LeRobotDatasetWriter.
        if np.issubdtype(arr.dtype, np.integer) or arr.max() > 1.0:
            finite = arr
            d_min, d_max = float(finite.min()), float(finite.max())
            if d_max > d_min:
                arr = (arr - d_min) / (d_max - d_min) * 255.0
            else:
                arr = np.zeros_like(arr)
        else:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
        depth_rgb = np.repeat(arr[:, :, None], 3, axis=2) if arr.ndim == 2 else arr
        return Image.fromarray(depth_rgb, mode="RGB").resize(
            (self.image_size[1], self.image_size[0]), Image.NEAREST
        )

    def _load_action(self, action):
        arr = self._to_numpy(action)
        if arr is None:
            return torch.zeros(self.action_dim)
        arr = arr.reshape(-1).astype(np.float32)
        if arr.shape[0] >= self.action_dim:
            arr = arr[: self.action_dim]
        else:
            pad = np.zeros(self.action_dim - arr.shape[0], dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        return torch.from_numpy(arr)

    # -------- sampling -----------------------------------------------------

    def _sample_frame_ids(self, episode_length):
        """Mirror LiberoRGBDDataset / RLinfLerobotDataset sampling policy."""
        if episode_length <= self.num_frames - 1:
            raise ValueError(
                f"[RLinfPickleDataset] episode length={episode_length} is too small "
                f"for num_frames={self.num_frames}"
            )
        if np.random.rand() < 0.95:
            if episode_length > 256:
                start_idx = np.random.randint(0, 250)
            else:
                start_idx = np.random.randint(0, episode_length - self.num_frames + 2)
            consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)
            frame_ids = np.concatenate([[0], consecutive_ids])
            frame_ids = np.clip(frame_ids, 0, episode_length - 1)
        else:
            frame_ids = np.array(
                [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8][: self.num_frames]
            )
            while len(frame_ids) < self.num_frames:
                frame_ids = np.append(frame_ids, frame_ids[-1] + 1)
                frame_ids[-1] = min(frame_ids[-1], episode_length - 1)
        return frame_ids

    def __len__(self):
        return len(self.episodes) * self.repeat

    def __getitem__(self, idx):
        # The slot may shift between calls because oversized-but-short episodes
        # are evicted lazily here when they sneak past the filesize pre-filter.
        # A small retry budget keeps any single sample from looping forever.
        attempts_remaining = 5
        while True:
            if not self.episodes:
                raise RuntimeError(
                    "[RLinfPickleDataset] all episodes filtered out at runtime; "
                    "rollout dump does not contain any window-length trajectories."
                )
            slot = idx % len(self.episodes)
            entry = self._get_episode(slot)
            episode_length = entry["length"]
            observations = entry["observations"]
            actions = entry["actions"]
            if episode_length >= self.window_size:
                break
            warnings.warn(
                f"[RLinfPickleDataset] runtime-drop short episode "
                f"file={os.path.basename(self.episodes[slot]['path'])} "
                f"length={episode_length} < window={self.window_size}"
            )
            # Evict from index and from cache so subsequent worker calls skip it.
            evicted = self.episodes.pop(slot)
            self._cache.pop(evicted["path"], None)
            if evicted["path"] in self._cache_order:
                self._cache_order.remove(evicted["path"])
            attempts_remaining -= 1
            if attempts_remaining <= 0:
                raise RuntimeError(
                    "[RLinfPickleDataset] exhausted retry budget while skipping "
                    "short episodes; check the rollout dump."
                )
            idx += 1  # try a neighboring slot

        if idx < 3:
            print(f"[DEBUG] RLinfPickle idx={idx}, slot={slot}, "
                  f"file={os.path.basename(self.episodes[slot]['path'])}, "
                  f"episode_length={episode_length}, num_frames={self.num_frames}")

        frame_ids = self._sample_frame_ids(episode_length)

        rgb_frames, depth_frames, action_list = [], [], []
        for frame_id in frame_ids:
            fid = int(frame_id)
            obs = observations[fid] if 0 <= fid < len(observations) else None
            rgb_frames.append(self._load_rgb(obs))
            depth_frames.append(self._load_depth(obs))
            action_list.append(self._load_action(actions[fid] if 0 <= fid < len(actions) else None))

        action_tensor = torch.stack(action_list)
        # Zero out the first (context) frame action; matches LiberoRGBDDataset.
        first_action = torch.zeros(self.action_dim)
        if self.action_dim >= 7:
            first_action[6] = -1.0
        action_tensor[0] = first_action
        assert torch.isfinite(action_tensor).all(), f"Action has NaN/Inf: {action_tensor}"
        return {
            "video": rgb_frames,
            "reference_image": [rgb_frames[0]],
            "action": action_tensor,
            "depth": depth_frames,
        }

    def load_episode_sequential(self, episode_idx, length=None):
        """Sequential loader matching ``LiberoRGBDDataset.load_episode_sequential``.

        Returns ``(rgb_list, actions_np, depth_pil_list)`` for the first
        ``length`` frames of episode ``episode_idx``. ``length=None`` uses the
        full episode length recorded in the pickle.

        Short-episode handling mirrors __getitem__: if the file ends up too
        short (slipped past the size filter) we evict it and surface a clear
        error rather than silently bisecting.
        """
        if not self.episodes:
            raise RuntimeError("[RLinfPickleDataset] no episodes available")
        slot = episode_idx % len(self.episodes)
        entry = self._get_episode(slot)
        ep_len = entry["length"]
        if ep_len < self.window_size:
            evicted = self.episodes.pop(slot)
            self._cache.pop(evicted["path"], None)
            if evicted["path"] in self._cache_order:
                self._cache_order.remove(evicted["path"])
            raise RuntimeError(
                f"[RLinfPickleDataset] episode {os.path.basename(evicted['path'])} "
                f"length={ep_len} < window={self.window_size}; evicted, retry "
                f"with a different episode_idx."
            )
        n = ep_len if length is None else min(int(length), ep_len)
        observations = entry["observations"]
        actions = entry["actions"]
        rgb_list, depth_list, action_list = [], [], []
        for fid in range(n):
            obs = observations[fid] if 0 <= fid < len(observations) else None
            rgb_list.append(self._load_rgb(obs))
            depth_list.append(self._load_depth(obs))
            action_list.append(self._load_action(actions[fid] if 0 <= fid < len(actions) else None).numpy())
        return rgb_list, np.stack(action_list, axis=0), depth_list


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        print(f"model_paths: {model_paths}")
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        # ❗
        # pipe.scheduler.set_timesteps(1000, training=True, enhance5steps=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, save_full_state_dict=False):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.save_full_state_dict = save_full_state_dict
        self.num_steps = 0


    def _get_state_dict(self, accelerator, model):
        state_dict = accelerator.get_state_dict(model)
        if self.save_full_state_dict:
            if self.remove_prefix_in_ckpt is not None:
                state_dict = {
                    (k[len(self.remove_prefix_in_ckpt):] if k.startswith(self.remove_prefix_in_ckpt) else k): v
                    for k, v in state_dict.items()
                }
        else:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
        return self.state_dict_converter(state_dict)


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = self._get_state_dict(accelerator, model)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = self._get_state_dict(accelerator, model)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def _load_sequential_val_sample(val_dataset, episode_idx=0, length=None):
    """Return a deterministic sequential window from ``val_dataset``.

    The standard ``__getitem__`` builds samples as ``[frame_0, frame_start, ..., frame_start+N-2]``
    with a random ``start_idx`` — fine for training, but useless for the validation
    visualisation since the GT clip teleports from episode head to a random middle
    segment while the model only sees ``frame_0`` as anchor. The two streams therefore
    cannot line up. This helper bypasses that sampling and returns
    ``[frame_0, frame_1, ..., frame_{length-1}]``.

    Supports the two depth-capable dataset shapes used in this branch:
      - LiberoRGBDDataset:    episodes=(start, end, length), dataset[i] -> dict item
      - RLinfLerobotDataset:  episodes=(root_idx, ep_idx, length), _read_row(...) -> dict row
    Falls back to a numpy-seeded ``__getitem__`` for anything else.
    """
    length = length or getattr(val_dataset, "window_size", None) \
        or getattr(val_dataset, "num_frames", None) or 9

    # LiberoRGBDDataset shape: episodes -> (start, end, length); dataset[i] is an item dict.
    if (
        hasattr(val_dataset, "episodes")
        and hasattr(val_dataset, "dataset")
        and hasattr(val_dataset, "_load_rgb")
    ):
        ep = val_dataset.episodes[episode_idx]
        ep_start, ep_end = int(ep[0]), int(ep[1])
        n = min(int(length), ep_end - ep_start)
        rgb_list, action_list, depth_list = [], [], []
        for i in range(ep_start, ep_start + n):
            item = val_dataset.dataset[i]
            rgb_list.append(val_dataset._load_rgb(item))
            action_list.append(val_dataset._load_action(item))
            if hasattr(val_dataset, "_load_depth"):
                depth_list.append(val_dataset._load_depth(item))
        actions = torch.stack(action_list)
        action_dim = actions.shape[-1]
        reset = torch.zeros(action_dim, dtype=actions.dtype, device=actions.device)
        if action_dim >= 1:
            reset[-1] = -1
        actions[0] = reset
        out = {"video": rgb_list, "reference_image": [rgb_list[0]], "action": actions}
        if depth_list:
            out["depth"] = depth_list
        return out

    # RLinfLerobotDataset shape: episodes -> (root_idx, ep_idx, length); rows via _read_row.
    if (
        hasattr(val_dataset, "episodes")
        and hasattr(val_dataset, "_read_row")
        and hasattr(val_dataset, "_load_rgb")
    ):
        root_idx, ep_idx, ep_len = val_dataset.episodes[episode_idx]
        n = min(int(length), int(ep_len))
        rgb_list, depth_list, action_list = [], [], []
        for fid in range(n):
            row = val_dataset._read_row(root_idx, ep_idx, int(fid))
            rgb_list.append(val_dataset._load_rgb(row.get(val_dataset.RGB_KEY)))
            if hasattr(val_dataset, "_load_depth"):
                depth_list.append(val_dataset._load_depth(row.get(val_dataset.DEPTH_KEY)))
            action_list.append(val_dataset._load_action(row.get(val_dataset.ACTION_KEY)))
        actions = torch.stack(action_list)
        action_dim = actions.shape[-1]
        reset = torch.zeros(action_dim, dtype=actions.dtype, device=actions.device)
        if action_dim >= 1:
            reset[-1] = -1
        actions[0] = reset
        out = {"video": rgb_list, "reference_image": [rgb_list[0]], "action": actions}
        if depth_list:
            out["depth"] = depth_list
        return out

    # Fallback: at least make the random sampling deterministic across epochs.
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        return val_dataset[episode_idx]
    finally:
        np.random.set_state(rng_state)


def _save_val_reconstruction(accelerator, model, val_dataset, epoch_id, args):
    """Run inference on a fixed validation sample and save reconstructed videos.

    No metrics, no loss — the goal is to eyeball training quality across epochs.
    Only main process does the work; others wait at the outer barrier.
    """
    if not accelerator.is_main_process:
        return

    from ..data.video import save_video

    unwrapped = accelerator.unwrap_model(model)
    pipe = unwrapped.pipe

    out_dir = os.path.join(args.output_path, "val_samples", f"epoch-{epoch_id:04d}")
    os.makedirs(out_dir, exist_ok=True)

    # IMPORTANT: do NOT use ``val_dataset[0]`` here. Its __getitem__ samples
    # ``frame_ids = [0, random_start, random_start+1, ...]``, which makes
    # ``gt_rgb`` a jump from episode head to a random middle segment, while the
    # model is anchored at ``frame_0`` only — gt and gen end up uncomparable.
    sample = _load_sequential_val_sample(val_dataset, episode_idx=0)
    gt_rgb = sample["video"]
    actions_full = sample["action"]
    gt_depth_pil = sample.get("depth", None)

    context_frames = getattr(val_dataset, "context_frames", 5)
    window = len(gt_rgb)

    input_image = gt_rgb[0]
    input_image4 = [input_image] * 4
    input_depth4 = [gt_depth_pil[0]] * 4 if gt_depth_pil is not None else None

    idx = np.array([0] * context_frames + list(range(1, window)))
    idx = np.clip(idx, 0, len(actions_full) - 1)
    action_win = actions_full[idx].to(device=pipe.device, dtype=pipe.torch_dtype)
    action_win[0] = 0
    action_win[0, -1] = -1

    num_frames_pipe = window + 4

    pipe.dit.eval()
    if getattr(pipe, "depth_dit", None) is not None:
        pipe.depth_dit.eval()
    if getattr(pipe, "cross_branch_attentions", None) is not None:
        pipe.cross_branch_attentions.eval()

    h = getattr(args, "height", None) or 256
    w = getattr(args, "width", None) or 256

    with torch.no_grad():
        out_video = pipe(
            seed=0,
            tiled=False,
            input_image=input_image,
            input_image4=input_image4,
            input_depth4=input_depth4,
            action=action_win,
            height=h,
            width=w,
            num_frames=num_frames_pipe,
            num_inference_steps=5,
            cfg_scale=1.0,
            sigma_shift=5.0,
            bs_1=True,
        )[0]

    # First-chunk slicing convention (mirror eval_dual_branch_stream.py i==0):
    # keep frame 0 then the last (window-1) predicted frames → length = window.
    n = min(window, len(out_video))
    gen_rgb = [out_video[0]] + list(out_video[-(n - 1):]) if n > 1 else [out_video[0]]

    raw_depth = getattr(pipe, "_last_depth_frames", None)
    if raw_depth and isinstance(raw_depth[0], list):
        chunk_depth = raw_depth[0]
    else:
        chunk_depth = raw_depth
    gen_depth = None
    if chunk_depth:
        gen_depth = [chunk_depth[0]] + list(chunk_depth[-(n - 1):]) if n > 1 else [chunk_depth[0]]

    save_video(gen_rgb[:n], os.path.join(out_dir, "gen.mp4"), fps=10, quality=5)
    save_video(gt_rgb[:n], os.path.join(out_dir, "gt.mp4"), fps=10, quality=5)
    side = [Image.fromarray(np.concatenate([np.array(g), np.array(r)], axis=1))
            for g, r in zip(gen_rgb[:n], gt_rgb[:n])]
    save_video(side, os.path.join(out_dir, "compare.mp4"), fps=10, quality=5)

    if gen_depth is not None and gt_depth_pil is not None:
        def _to_rgb_gray(d):
            gray = np.array(d.convert("L"))
            return Image.fromarray(np.stack([gray, gray, gray], axis=-1))
        depth_pred_frames = [_to_rgb_gray(d) for d in gen_depth[:n]]
        depth_gt_frames = [_to_rgb_gray(d) for d in gt_depth_pil[:n]]
        save_video(depth_pred_frames, os.path.join(out_dir, "depth_pred.mp4"), fps=10, quality=5)
        save_video(depth_gt_frames, os.path.join(out_dir, "depth_gt.mp4"), fps=10, quality=5)
        m2 = min(len(depth_pred_frames), len(depth_gt_frames))
        depth_cmp = [Image.fromarray(np.concatenate([np.array(dp), np.array(dg)], axis=1))
                     for dp, dg in zip(depth_pred_frames[:m2], depth_gt_frames[:m2])]
        save_video(depth_cmp, os.path.join(out_dir, "depth_compare.mp4"), fps=10, quality=5)

    print(f"[val] Saved reconstruction for epoch {epoch_id} → {out_dir}")


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    save_epochs: int = 1,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        save_epochs = args.save_epochs
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        ###验证集
        val_interval = args.val_interval # 5
    writer = SummaryWriter(log_dir=os.path.join(args.output_path, "tensorboard") if args is not None else "./tensorboard")

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, val_dataloader, scheduler)

    global_step = 0
    epoch_loss = 0.0
    epoch_steps = 0

    for epoch_id in range(num_epochs):

        epoch_loss = 0.0
        epoch_steps = 0

        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if getattr(dataset, 'load_from_cache', False):
                    out = model({}, inputs=data)
                else:
                    out = model(data)
                if isinstance(out, dict):
                    loss = out["loss"]
                    aux = {k: v.item() for k, v in out.items() if k != "loss"}
                else:
                    loss = out
                    aux = {}
                loss_str = " ".join(f"{k}: {v:.4f}" for k, v in aux.items())
                print(f'Epoch {epoch_id} Step {global_step} Loss: {loss.item():.4f}' + (f'  [{loss_str}]' if aux else ''))
                accelerator.backward(loss)

                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

                if accelerator.is_main_process:
                    writer.add_scalar("Loss/step", loss.item(), global_step)
                    for k, v in aux.items():
                        writer.add_scalar(f"Loss/{k}_step", v, global_step)
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # to avoid the too long epoch 
                if epoch_steps > 500:
                    break
        if accelerator.is_main_process and epoch_steps > 0:
            writer.add_scalar("Loss/epoch", epoch_loss / epoch_steps, epoch_id)

        if val_dataset is not None and (epoch_id + 1) % val_interval == 0:
            model.eval()
            if accelerator.is_main_process:
                print(f"\nRunning validation reconstruction for epoch {epoch_id}...")
            try:
                _save_val_reconstruction(accelerator, model, val_dataset, epoch_id, args)
            except Exception as e:
                if accelerator.is_main_process:
                    import traceback
                    print(f"[val] Reconstruction failed for epoch {epoch_id}: {e}")
                    traceback.print_exc()
            accelerator.wait_for_everyone()
            model.train()
            # --------------------
        if save_steps is None and (epoch_id + 1) % save_epochs == 0:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    writer.close()


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Model ID with origin paths to the audio processor config, e.g., Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--save_epochs", type=int, default=1, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser
