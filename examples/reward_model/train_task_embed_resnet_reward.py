import argparse
import importlib.util
import json
import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from diffsynth.models.reward_model import TaskEmbedResnetRewModel, instruction_to_task_id
except Exception:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    reward_model_path = os.path.join(repo_root, "diffsynth", "models", "reward_model.py")
    spec = importlib.util.spec_from_file_location("reward_model", reward_model_path)
    reward_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_model)
    TaskEmbedResnetRewModel = reward_model.TaskEmbedResnetRewModel
    instruction_to_task_id = reward_model.instruction_to_task_id


IMAGE_KEYS = (
    "observation.images.image",
    "observation.image",
    "main_images",
    "image",
    "rgb",
    "obs",
    "observation",
)
DEPTH_KEYS = (
    "observation.images.image_depth",
    "observation.depth",
    "main_depths",
    "main_depth",
    "depth",
)
INSTRUCTION_KEYS = (
    "task",
    "task_descriptions",
    "instruction",
    "language_instruction",
    "natural_language_instruction",
    "prompt",
)
TASK_ID_KEYS = ("task_id", "task_index", "task_idx")
FRAME_LABEL_KEYS = ("reward", "success", "is_success", "done")
EPISODE_LABEL_KEYS = ("success", "is_success", "episode_success", "successes")
FRAME_LIST_KEYS = ("observations", "frames", "steps", "transitions", "data")
KNOWN_BAD_PICKLES = {
    "rank_5_env_3_episode_23_step_12288_fail.pkl",
    "rank_7_env_1_episode_23_step_12288_fail.pkl",
}


@dataclass(frozen=True)
class RewardSample:
    episode_idx: int
    frame_idx: int
    label: float
    source: str = ""


def get_value(obj: Any, keys: Sequence[str], default=None):
    if not isinstance(obj, dict):
        return default
    for key in keys:
        if key in obj:
            return obj[key]
        cur = obj
        found = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                found = False
                break
        if found:
            return cur
    return default


def is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray, torch.Tensor))


def sequence_len(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.shape[0])
    return len(value)


def sequence_item(value: Any, index: int):
    if torch.is_tensor(value):
        return value[index]
    return value[index]


def scalar_value(value: Any):
    if torch.is_tensor(value):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.reshape(-1)[0]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def load_pickle_like(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def discover_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for name in names:
            if name.endswith((".pkl", ".pickle", ".pt", ".pth")):
                files.append(os.path.join(root, name))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No pickle-like files found under {path}")
    return files


def split_items(items: List[Any], split: str, val_ratio: float, seed: int) -> List[Any]:
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * val_ratio)) if len(indices) > 1 else 0
    if split == "train":
        keep = indices[:-val_count] if val_count else indices
    elif split in ("val", "validation"):
        keep = indices[-val_count:] if val_count else indices
    else:
        raise ValueError(f"Unknown split: {split}")
    keep.sort()
    return [items[i] for i in keep]


def coerce_episodes(obj: Any) -> List[Any]:
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return []
    if isinstance(obj, dict):
        for key in ("episodes", "trajectories", "rollouts"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        return [obj]
    if isinstance(obj, list):
        if not obj:
            return []
        if all(isinstance(item, dict) for item in obj):
            has_frame_keys = any(get_value(obj[0], IMAGE_KEYS) is not None for _ in [0])
            if has_frame_keys:
                return [obj]
        return obj
    return [obj]


def episode_frames(episode: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(episode, list) and all(isinstance(x, dict) for x in episode):
        return episode
    if isinstance(episode, dict):
        frames = get_value(episode, FRAME_LIST_KEYS)
        if isinstance(frames, list) and all(isinstance(x, dict) for x in frames):
            return frames
    return None


def episode_length(episode: Any) -> int:
    frames = episode_frames(episode)
    if frames is not None:
        return len(frames)
    image_seq = get_value(episode, IMAGE_KEYS)
    if image_seq is None or not is_sequence(image_seq):
        raise ValueError(f"Cannot infer episode length. Top-level keys: {schema_keys(episode)}")
    return sequence_len(image_seq)


def schema_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return sorted(list(obj.keys()))
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return {"list_len": len(obj), "frame_keys": sorted(list(obj[0].keys()))}
    return type(obj).__name__


class RLinfPickleRewardDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        image_size: Tuple[int, int] = (256, 256),
        task_suite_name: str = "libero_goal",
        input_modalities: str = "rgb",
        positive_per_success_episode: int = 8,
        negative_per_success_episode: int = 8,
        negative_per_fail_episode: int = 16,
        balance_samples: bool = True,
        lazy_files: bool = True,
        samples_per_epoch: int = 20000,
        episode_frames_per_item: int = 32,
        min_file_size_mb: float = 0.0,
        max_episodes: Optional[int] = None,
        crop_rgb_from_stitched: str = "none",
    ):
        self.data_path = data_path
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        self.image_size = image_size
        self.task_suite_name = task_suite_name
        self.input_modalities = input_modalities
        self.positive_per_success_episode = positive_per_success_episode
        self.negative_per_success_episode = negative_per_success_episode
        self.negative_per_fail_episode = negative_per_fail_episode
        self.balance_samples = balance_samples
        self.lazy_files = lazy_files
        self.samples_per_epoch = samples_per_epoch
        self.episode_frames_per_item = episode_frames_per_item
        self.min_file_size_mb = min_file_size_mb
        self.crop_rgb_from_stitched = crop_rgb_from_stitched

        if self.lazy_files and os.path.isdir(data_path):
            all_files = [
                p for p in discover_files(data_path)
                if "_success" in os.path.basename(p) or "_fail" in os.path.basename(p)
            ]
            before_bad_filter = len(all_files)
            all_files = [p for p in all_files if os.path.basename(p) not in KNOWN_BAD_PICKLES]
            if before_bad_filter != len(all_files):
                print(
                    f"[RLinfPickleRewardDataset] skipped known bad pickle files: "
                    f"{before_bad_filter - len(all_files)}"
                )
            if self.min_file_size_mb > 0:
                min_bytes = int(self.min_file_size_mb * 1024 * 1024)
                before = len(all_files)
                all_files = [p for p in all_files if os.path.getsize(p) >= min_bytes]
                print(
                    f"[RLinfPickleRewardDataset] filtered small files: "
                    f"{before - len(all_files)} skipped below {self.min_file_size_mb} MB"
                )
            self.files = split_items(all_files, split, val_ratio, seed)
            if max_episodes is not None:
                self.files = self.files[:max_episodes]
            self.success_files = [p for p in self.files if "_success" in os.path.basename(p)]
            self.fail_files = [p for p in self.files if "_fail" in os.path.basename(p)]
            if not self.success_files and not self.fail_files:
                raise ValueError("Lazy mode requires '_success'/'_fail' in pickle file names.")
            print(
                f"[RLinfPickleRewardDataset] lazy split={split}, files={len(self.files)}, "
                f"success_files={len(self.success_files)}, fail_files={len(self.fail_files)}, "
                f"samples_per_epoch={self.samples_per_epoch}"
            )
            return

        all_episodes = []
        for file_path in discover_files(data_path):
            obj = load_pickle_like(file_path)
            all_episodes.extend(coerce_episodes(obj))
            if max_episodes is not None and len(all_episodes) >= max_episodes:
                all_episodes = all_episodes[:max_episodes]
                break
        if not all_episodes:
            raise ValueError(f"No episodes loaded from {data_path}")

        self.episodes = self._split_episodes(all_episodes)
        self.samples: List[RewardSample] = []
        for ep_idx, episode in enumerate(self.episodes):
            self.samples.extend(self._index_episode(ep_idx, episode))
        if self.balance_samples:
            self.samples = self._balance_samples(self.samples)
        if not self.samples:
            raise ValueError("No reward samples were built. Check image/label fields in the pickle schema.")

        pos = sum(1 for s in self.samples if s.label > 0.5)
        neg = len(self.samples) - pos
        sources = {}
        for sample in self.samples:
            sources[sample.source] = sources.get(sample.source, 0) + 1
        print(
            f"[RLinfPickleRewardDataset] split={split}, episodes={len(self.episodes)}, "
            f"samples={len(self.samples)}, pos={pos}, neg={neg}, sources={sources}"
        )

    def _split_episodes(self, episodes: List[Any]) -> List[Any]:
        return split_items(episodes, self.split, self.val_ratio, self.seed)

    def _index_episode(self, ep_idx: int, episode: Any) -> List[RewardSample]:
        length = episode_length(episode)
        if length <= 0:
            return []

        ep_success = self._episode_success(episode)
        rng = random.Random(self.seed + ep_idx)
        labels = self._success_once_frame_labels(episode, length)
        if labels is None:
            labels = self._reward_frame_labels(episode, length)

        if labels is not None and any(labels):
            pos_pool = [i for i, label in enumerate(labels) if label]
            neg_pool = [i for i, label in enumerate(labels) if not label]
            rng.shuffle(pos_pool)
            rng.shuffle(neg_pool)
            pos = [
                RewardSample(ep_idx, i, 1.0, "success_pos")
                for i in pos_pool[: min(self.positive_per_success_episode, len(pos_pool))]
            ]
            neg = [
                RewardSample(ep_idx, i, 0.0, "success_neg")
                for i in neg_pool[: min(self.negative_per_success_episode, len(neg_pool))]
            ]
            return pos + neg

        if ep_success:
            first_pos = max(0, length - self.positive_per_success_episode)
            samples = [RewardSample(ep_idx, i, 1.0, "success_pos_fallback") for i in range(first_pos, length)]
            neg_pool = list(range(0, max(1, first_pos)))
            neg_limit = self.negative_per_success_episode
        else:
            samples = []
            neg_pool = list(range(length))
            neg_limit = self.negative_per_fail_episode

        rng.shuffle(neg_pool)
        source = "success_neg_fallback" if ep_success else "fail_neg"
        for i in neg_pool[: min(neg_limit, len(neg_pool))]:
            samples.append(RewardSample(ep_idx, i, 0.0, source))
        return samples

    def _success_once_frame_labels(self, episode: Any, length: int) -> Optional[List[bool]]:
        infos = get_value(episode, ("infos",))
        if infos is None or not is_sequence(infos) or sequence_len(infos) != length:
            return None
        labels = []
        found = False
        for i in range(length):
            info = sequence_item(infos, i)
            episode_info = info.get("episode") if isinstance(info, dict) else None
            if isinstance(episode_info, dict) and "success_once" in episode_info:
                found = True
                labels.append(bool(scalar_value(episode_info["success_once"])))
            else:
                labels.append(False)
        return labels if found else None

    def _reward_frame_labels(self, episode: Any, length: int) -> Optional[List[bool]]:
        frames = episode_frames(episode)
        if frames is not None:
            values = [get_value(frame, FRAME_LABEL_KEYS) for frame in frames]
            if any(v is not None for v in values):
                return [float(scalar_value(v) if v is not None else 0.0) > 0.0 for v in values]

        seq = get_value(episode, FRAME_LABEL_KEYS)
        if seq is not None and is_sequence(seq) and sequence_len(seq) == length:
            return [float(scalar_value(sequence_item(seq, i))) > 0.0 for i in range(length)]
        return None

    def _balance_samples(self, samples: List[RewardSample]) -> List[RewardSample]:
        success_pos = [s for s in samples if s.source.startswith("success_pos")]
        success_neg = [s for s in samples if s.source.startswith("success_neg")]
        fail_neg = [s for s in samples if s.source == "fail_neg"]
        if not success_pos or not success_neg or not fail_neg:
            return samples

        rng = random.Random(self.seed + (0 if self.split == "train" else 100000))
        for pool in (success_pos, success_neg, fail_neg):
            rng.shuffle(pool)

        per_success_class = min(len(success_pos), len(success_neg), len(fail_neg) // 2)
        if per_success_class <= 0:
            return samples
        fail_count = min(len(fail_neg), per_success_class * 2)
        balanced = success_pos[:per_success_class] + success_neg[:per_success_class] + fail_neg[:fail_count]
        rng.shuffle(balanced)
        return balanced

    def _episode_success(self, episode: Any) -> bool:
        value = get_value(episode, EPISODE_LABEL_KEYS)
        if value is None:
            return False
        if is_sequence(value):
            if sequence_len(value) == 0:
                return False
            value = sequence_item(value, sequence_len(value) - 1)
        return bool(scalar_value(value))

    def _frame(self, episode: Any, frame_idx: int) -> Dict[str, Any]:
        frames = episode_frames(episode)
        if frames is not None:
            return frames[frame_idx]
        return episode

    def _image(self, episode: Any, frame_idx: int):
        frame = self._frame(episode, frame_idx)
        image = get_value(frame, IMAGE_KEYS)
        if image is not None:
            return image
        image_seq = get_value(episode, IMAGE_KEYS)
        if image_seq is None:
            raise ValueError(f"No image field found. Episode schema: {schema_keys(episode)}")
        return sequence_item(image_seq, frame_idx)

    def _depth(self, episode: Any, frame_idx: int):
        frame = self._frame(episode, frame_idx)
        depth = get_value(frame, DEPTH_KEYS)
        if depth is not None:
            return depth
        depths = get_value(episode, ("depths.main_depth", "depths.main_depths", "main_depths", "main_depth", "depth"))
        if isinstance(depths, dict):
            depths = get_value(depths, ("main_depth", "main_depths", "depth"))
        if depths is None:
            raise ValueError(f"No depth field found. Episode schema: {schema_keys(episode)}")
        return sequence_item(depths, frame_idx)

    def _instruction_or_task_id(self, episode: Any, frame_idx: int):
        frame = self._frame(episode, frame_idx)
        task_id = get_value(frame, TASK_ID_KEYS)
        if task_id is None:
            task_id = get_value(episode, TASK_ID_KEYS)
        if task_id is not None:
            if torch.is_tensor(task_id):
                task_id = task_id[frame_idx] if task_id.ndim > 0 and task_id.shape[0] > frame_idx else task_id
                task_id = task_id.item()
            if isinstance(task_id, np.ndarray):
                task_id = task_id.item() if task_id.ndim == 0 else task_id[frame_idx if task_id.shape[0] > frame_idx else 0]
            return int(task_id)

        instruction = get_value(frame, INSTRUCTION_KEYS)
        if instruction is None:
            instruction = get_value(episode, INSTRUCTION_KEYS)
        if instruction is None:
            raise ValueError(f"No instruction/task_id field found. Episode schema: {schema_keys(episode)}")
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        if hasattr(instruction, "item"):
            instruction = instruction.item()
        return instruction

    def _to_tensor(self, image: Any) -> torch.Tensor:
        if torch.is_tensor(image):
            arr = image.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4, 6):
                arr = arr[:3].permute(1, 2, 0)
            arr = arr.numpy()
        elif isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.array(image)

        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4, 6) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]

        if self.crop_rgb_from_stitched != "none" and arr.ndim == 3:
            mid = arr.shape[1] // 2
            arr = arr[:, :mid] if self.crop_rgb_from_stitched == "left" else arr[:, mid:]

        arr = arr.astype(np.float32)
        if arr.max() > 2.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        pil = Image.fromarray((arr * 255.0).astype(np.uint8)).resize(
            (self.image_size[1], self.image_size[0]), Image.BILINEAR
        )
        arr = np.array(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor * 2.0 - 1.0

    def _depth_to_tensor(self, depth: Any) -> torch.Tensor:
        if torch.is_tensor(depth):
            arr = depth.detach().cpu().numpy()
        elif isinstance(depth, Image.Image):
            arr = np.array(depth)
        else:
            arr = np.array(depth)

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = arr[0]
            else:
                arr = arr[:, :, 0]

        was_integer = np.issubdtype(arr.dtype, np.integer)
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size == 0:
            arr = np.zeros(tuple(self.image_size), dtype=np.float32)

        if was_integer or arr.max() > 1.0:
            d_min, d_max = float(arr.min()), float(arr.max())
            if d_max > d_min:
                arr = (arr - d_min) / (d_max - d_min) * 255.0
            else:
                arr = np.zeros_like(arr)
        else:
            arr = arr * 255.0

        arr = arr.clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L").resize(
            (self.image_size[1], self.image_size[0]), Image.NEAREST
        )
        arr = np.array(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def _obs_tensor(self, episode: Any, frame_idx: int) -> torch.Tensor:
        rgb = self._to_tensor(self._image(episode, frame_idx))
        if self.input_modalities == "rgb":
            return rgb
        if self.input_modalities == "rgbd":
            depth = self._depth_to_tensor(self._depth(episode, frame_idx))
            return torch.cat([rgb, depth], dim=0)
        raise ValueError(f"Unknown input_modalities: {self.input_modalities}")

    def __len__(self):
        if self.lazy_files:
            return self.samples_per_epoch
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.lazy_files:
            return self._getitem_lazy(index)
        sample = self.samples[index]
        episode = self.episodes[sample.episode_idx]
        return self._make_item(episode, sample.frame_idx, sample.label)

    def _make_item(self, episode: Any, frame_idx: int, label: float) -> Dict[str, Any]:
        task = self._instruction_or_task_id(episode, frame_idx)
        if isinstance(task, str):
            task_id = instruction_to_task_id(task, self.task_suite_name).item()
            instruction = task
        else:
            task_id = int(task)
            instruction = ""
        return {
            "obs": self._obs_tensor(episode, frame_idx),
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "label": torch.tensor([label], dtype=torch.float32),
            "instruction": instruction,
        }

    def _getitem_lazy(self, index: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + index + (0 if self.split == "train" else 1000000))
        frames_per_item = max(1, self.episode_frames_per_item)

        last_error = None
        for _ in range(20):
            try:
                choose_fail = bool(self.fail_files) and (not self.success_files or rng.random() < 0.5)
                if choose_fail:
                    path = rng.choice(self.fail_files)
                    episode = coerce_episodes(load_pickle_like(path))[0]
                    length = episode_length(episode)
                    frame_indices = [rng.randrange(length) for _ in range(frames_per_item)]
                    return self._make_items(episode, frame_indices, [0.0] * len(frame_indices))

                path = rng.choice(self.success_files)
                episode = coerce_episodes(load_pickle_like(path))[0]
                length = episode_length(episode)
                labels = self._success_once_frame_labels(episode, length)
                if labels is None:
                    labels = self._reward_frame_labels(episode, length)
                pos_pool = [i for i, label in enumerate(labels or []) if label]
                neg_pool = [i for i, label in enumerate(labels or []) if not label]

                frame_indices = []
                label_values = []
                n_pos = frames_per_item // 2
                n_neg = frames_per_item - n_pos
                if pos_pool:
                    frame_indices.extend(rng.choice(pos_pool) for _ in range(n_pos))
                    label_values.extend([1.0] * n_pos)
                if neg_pool:
                    frame_indices.extend(rng.choice(neg_pool) for _ in range(n_neg))
                    label_values.extend([0.0] * n_neg)
                while len(frame_indices) < frames_per_item:
                    frame_indices.append(rng.randrange(length))
                    label_values.append(0.0)
                paired = list(zip(frame_indices, label_values))
                rng.shuffle(paired)
                frame_indices, label_values = zip(*paired)
                return self._make_items(episode, list(frame_indices), list(label_values))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to sample a valid lazy item after retries: {last_error}")

    def _make_items(self, episode: Any, frame_indices: List[int], labels: List[float]) -> Dict[str, Any]:
        items = [self._make_item(episode, frame_idx, label) for frame_idx, label in zip(frame_indices, labels)]
        return {
            "obs": torch.stack([item["obs"] for item in items]),
            "task_id": torch.stack([item["task_id"] for item in items]),
            "label": torch.stack([item["label"] for item in items]),
            "instruction": [item["instruction"] for item in items],
        }

    def describe_schema(self, max_items: int = 1):
        desc = []
        if self.lazy_files:
            for path in self.files[:max_items]:
                episode = coerce_episodes(load_pickle_like(path))[0]
                desc.append({"file": os.path.basename(path), "episode": schema_keys(episode), "length": episode_length(episode)})
            return desc
        for episode in self.episodes[:max_items]:
            desc.append({"episode": schema_keys(episode), "length": episode_length(episode)})
        return desc


def collate_reward(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if batch and batch[0]["obs"].ndim == 4:
        return {
            "obs": torch.cat([x["obs"] for x in batch], dim=0),
            "task_id": torch.cat([x["task_id"] for x in batch], dim=0),
            "label": torch.cat([x["label"] for x in batch], dim=0),
            "instruction": sum((x["instruction"] for x in batch), []),
        }
    return {
        "obs": torch.stack([x["obs"] for x in batch]),
        "task_id": torch.stack([x["task_id"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
        "instruction": [x["instruction"] for x in batch],
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total = correct = tp = fp = fn = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        obs = batch["obs"].to(device)
        task_id = batch["task_id"].to(device)
        label = batch["label"].to(device)
        logits = model.forward_logits(obs, task_id=task_id)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        prob = torch.sigmoid(logits)
        pred = prob >= 0.5
        truth = label >= 0.5
        total_loss += loss.item() * obs.shape[0]
        total += obs.shape[0]
        correct += (pred == truth).sum().item()
        tp += (pred & truth).sum().item()
        fp += (pred & ~truth).sum().item()
        fn += (~pred & truth).sum().item()
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "loss": total_loss / max(1.0, total),
        "acc": correct / max(1.0, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--task_embed_dim", type=int, default=64)
    parser.add_argument("--input_channels", type=int, default=None)
    parser.add_argument("--input_modalities", choices=["rgb", "rgbd"], default="rgb")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", default="auto", help="'auto', 'none', or a numeric BCE positive-class weight.")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive_per_success_episode", type=int, default=8)
    parser.add_argument("--negative_per_success_episode", type=int, default=8)
    parser.add_argument("--negative_per_fail_episode", type=int, default=16)
    parser.add_argument("--no_balance_samples", action="store_true")
    parser.add_argument("--no_lazy_files", action="store_true")
    parser.add_argument("--samples_per_epoch", type=int, default=20000)
    parser.add_argument("--episode_frames_per_item", type=int, default=32)
    parser.add_argument("--min_file_size_mb", type=float, default=0.0)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--crop_rgb_from_stitched", choices=["none", "left", "right"], default="none")
    parser.add_argument("--inspect_only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_set = RLinfPickleRewardDataset(
        args.data_path,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_size=tuple(args.image_size),
        task_suite_name=args.task_suite_name,
        input_modalities=args.input_modalities,
        positive_per_success_episode=args.positive_per_success_episode,
        negative_per_success_episode=args.negative_per_success_episode,
        negative_per_fail_episode=args.negative_per_fail_episode,
        balance_samples=not args.no_balance_samples,
        lazy_files=not args.no_lazy_files,
        samples_per_epoch=args.samples_per_epoch,
        episode_frames_per_item=args.episode_frames_per_item,
        min_file_size_mb=args.min_file_size_mb,
        max_episodes=args.max_episodes,
        crop_rgb_from_stitched=args.crop_rgb_from_stitched,
    )
    val_set = RLinfPickleRewardDataset(
        args.data_path,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_size=tuple(args.image_size),
        task_suite_name=args.task_suite_name,
        input_modalities=args.input_modalities,
        positive_per_success_episode=args.positive_per_success_episode,
        negative_per_success_episode=args.negative_per_success_episode,
        negative_per_fail_episode=args.negative_per_fail_episode,
        balance_samples=not args.no_balance_samples,
        lazy_files=not args.no_lazy_files,
        samples_per_epoch=max(args.batch_size, args.samples_per_epoch // 10),
        episode_frames_per_item=args.episode_frames_per_item,
        min_file_size_mb=args.min_file_size_mb,
        max_episodes=args.max_episodes,
        crop_rgb_from_stitched=args.crop_rgb_from_stitched,
    )
    if args.inspect_only:
        print(json.dumps(train_set.describe_schema(), indent=2, default=str))
        return

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_reward,
        pin_memory=True,
        drop_last=len(train_set) > args.batch_size,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_reward,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = args.input_channels
    if input_channels is None:
        input_channels = 4 if args.input_modalities == "rgbd" else 3
    print(f"[train] input_modalities={args.input_modalities} input_channels={input_channels}")
    model = TaskEmbedResnetRewModel(
        checkpoint_path=args.checkpoint_path,
        num_tasks=args.num_tasks,
        task_embed_dim=args.task_embed_dim,
        task_suite_name=args.task_suite_name,
        input_channels=input_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.pos_weight == "auto":
        if getattr(train_set, "lazy_files", False):
            pos, neg = 1, 3
        else:
            pos = sum(1 for sample in train_set.samples if sample.label > 0.5)
            neg = len(train_set.samples) - pos
        pos_weight = torch.tensor([neg / max(1, pos)], device=device)
    elif args.pos_weight == "none":
        pos_weight = None
    else:
        pos_weight = torch.tensor([float(args.pos_weight)], device=device)
    print(f"[train] pos_weight={None if pos_weight is None else pos_weight.item()}")

    best_f1 = -1.0
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            obs = batch["obs"].to(device)
            task_id = batch["task_id"].to(device)
            label = batch["label"].to(device)
            logits = model.forward_logits(obs, task_id=task_id)
            loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * obs.shape[0]
            seen += obs.shape[0]

        metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / max(1, seen)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={metrics['loss']:.6f} "
            f"acc={metrics['acc']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

        save_checkpoint(os.path.join(args.output_path, "last.pth"), model, optimizer, epoch, metrics, args)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(os.path.join(args.output_path, "best.pth"), model, optimizer, epoch, metrics, args)


if __name__ == "__main__":
    main()
