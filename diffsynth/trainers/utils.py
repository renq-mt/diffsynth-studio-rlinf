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
        max_depth: float = 5.0,
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
        self.max_depth = max_depth
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
        """Load depth map as PIL Image (uint8 [0, 255]) - grayscale."""
        import numpy as np
        depth = item.get(self.DEPTH_KEY)
        if depth is None:
            # Return white depth map (max depth)
            return Image.new("L", (self.image_size[1], self.image_size[0]), 255)

        if isinstance(depth, Image.Image):
            # Convert PIL Image to array, convert to grayscale if needed
            depth = np.array(depth).astype(np.float32)
        else:
            depth = np.array(depth).astype(np.float32)

        # Handle multi-channel depth (e.g., RGB-encoded depth)
        # Take the first channel or convert to grayscale
        if depth.ndim == 3:
            # If depth has shape (H, W, C), take first channel or use grayscale conversion
            if depth.shape[2] == 1:
                depth = depth[:, :, 0]
            elif depth.shape[2] == 3:
                # RGB-encoded depth: use standard grayscale conversion
                # Or just take the red channel if it's single-channel data encoded as RGB
                depth = depth[:, :, 0]  # Take first channel
            else:
                depth = depth[:, :, 0]  # Take first channel

        # Now depth should be 2D (H, W)
        assert depth.ndim == 2, f"Depth should be 2D after processing, got shape {depth.shape}"

        # Normalize depth to [0, 1]
        if depth.max() <= 1.0 and depth.max() > 0:
            # Already normalized
            depth_normalized = depth
        elif depth.max() > 1.0:
            # Raw depth values, normalize by max_depth
            depth_normalized = np.clip(depth / self.max_depth, 0.0, 1.0)
        else:
            # All zeros
            depth_normalized = depth

        # Convert to uint8 [0, 255] PIL Image (explicitly grayscale)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        return Image.fromarray(depth_uint8, mode='L').resize(
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

        # Convert depth PIL Images to tensor
        import torchvision.transforms.functional as TF
        depth_tensors = [TF.to_tensor(d) for d in depth_frames]  # Each is [1, H, W] in [0, 1]
        depth_tensor = torch.stack(depth_tensors)  # (num_frames, 1, H, W)

        # Sanity checks to ensure data is valid
        assert torch.isfinite(action_tensor).all(), f"Action has NaN/Inf: {action_tensor}"
        assert torch.isfinite(depth_tensor).all(), f"Depth has NaN/Inf: min={depth_tensor.min()}, max={depth_tensor.max()}"
        assert depth_tensor.min() >= 0.0 and depth_tensor.max() <= 1.0, f"Depth out of [0,1] range: min={depth_tensor.min()}, max={depth_tensor.max()}"

        return {
            "video": video_list,
            "reference_image": [video_list[0]],
            "action": action_tensor,
            "depth": depth_tensor,  # (num_frames, 1, H, W) in [0,1]
        }
    
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
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


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
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                print(f'Epoch {epoch_id} Step {global_step} Loss: {loss.item()}')
                accelerator.backward(loss)

                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

                if accelerator.is_main_process:
                    writer.add_scalar("Loss/step", loss.item(), global_step)
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # to avoid the too long epoch 
                if epoch_steps > 500:
                    break
        if accelerator.is_main_process and epoch_steps > 0:
            writer.add_scalar("Loss/epoch", epoch_loss / epoch_steps, epoch_id)

        if val_dataloader is not None and (epoch_id + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            if accelerator.is_main_process:
                print(f"\nRunning validation for epoch {epoch_id}...")

            for data in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    if getattr(val_dataset, 'load_from_cache', False):
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    

                    avg_loss = accelerator.gather(loss).mean().item()
                    val_loss += avg_loss
                    val_steps += 1

                    if val_steps > 500:
                        break
            
            if val_steps > 0:
                avg_val_loss = val_loss / val_steps
                if accelerator.is_main_process:
                    writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch_id)
                    print(f"Epoch {epoch_id} Validation Loss: {avg_val_loss}")
            
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
