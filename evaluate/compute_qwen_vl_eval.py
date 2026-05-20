import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from PIL import Image
from tqdm import tqdm

try:
    from .common import VIDEO_EXTS, ensure_dir, iter_video_pairs, load_json, metric_output_path, save_json, summarize_values
except ImportError:
    from common import VIDEO_EXTS, ensure_dir, iter_video_pairs, load_json, metric_output_path, save_json, summarize_values


DEFAULT_MODEL_PATH = os.environ.get("QWEN3_VL_MODEL", "")
METRIC_KEYS = ("Interaction_Quality", "Perspectivity", "Instruction_Following")
NO_INSTRUCTION_METRIC_KEYS = ("Interaction_Quality", "Perspectivity")
INSTRUCTION_KEYS = (
    "instruction",
    "language_instruction",
    "task_instruction",
    "task",
    "prompt",
    "caption",
)


def _as_instruction(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _as_instruction(value[0])
    return ""


def _instruction_from_row(row: Dict) -> str:
    for key in INSTRUCTION_KEYS:
        text = _as_instruction(row.get(key))
        if text:
            return text
    return ""


def _candidate_video_names(row: Dict) -> List[str]:
    names = []
    for key in ("video", "video_path", "gen_video", "pred_video", "gt_video", "gt_path", "path"):
        value = row.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            names.append(path.name)
            names.append(f"{path.stem}.mp4")
    for key in ("sample_id", "episode", "id"):
        value = row.get(key)
        if value is not None:
            text = str(value)
            names.extend([text, f"{text}.mp4"])
    return list(dict.fromkeys(names))


def load_instruction_map(instruction_json: Optional[Path]) -> Dict[str, str]:
    if instruction_json is None or not instruction_json.exists():
        return {}
    data = load_json(instruction_json)
    rows = data.get("episodes", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}

    instruction_map: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        instruction = _instruction_from_row(row)
        if not instruction:
            continue
        for name in _candidate_video_names(row):
            instruction_map[name] = instruction
    return instruction_map


def load_manifest_instruction_map(pred_root: Path) -> Dict[str, str]:
    return load_instruction_map(pred_root / "manifest.json")


def list_pred_videos(pred_root: Path, true_root: Optional[Path] = None) -> List[Tuple[str, Path]]:
    pairs = iter_video_pairs(pred_root, true_root)
    if pairs:
        return [(sample_id, pred) for sample_id, pred, _ in pairs]

    videos: List[Tuple[str, Path]] = []
    for path in sorted(pred_root.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            videos.append((path.stem, path))
        elif path.is_dir():
            nested = path / "gen.mp4"
            if nested.exists():
                videos.append((path.name, nested))
    return videos


def sample_video_frames(
    video_path: Path,
    num_frames: int = 16,
    skip_first: int = 0,
    start_fraction: float = 0.0,
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fraction_start = int(total_frames * max(0.0, min(1.0, start_fraction)))
        start_idx = max(skip_first, fraction_start)
        if total_frames <= start_idx:
            return frames

        usable_frames = total_frames - start_idx
        if usable_frames <= num_frames:
            indices = list(range(start_idx, total_frames))
        else:
            indices = [
                start_idx + int(round(i * (usable_frames - 1) / max(1, num_frames - 1)))
                for i in range(num_frames)
            ]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    finally:
        cap.release()

    return frames[:num_frames]


def build_action_quality_prompt(instruction_text: str = "", *, include_instruction_following: bool = True) -> str:
    prompt = ""
    if include_instruction_following and instruction_text:
        prompt += f"\n**VIDEO INSTRUCTION:** {instruction_text}\n"
    prompt += """
You are an expert evaluator for robot interaction videos. You are evaluating videos generated for embodied AI manipulation scenarios, specifically focusing on robotic arms interacting with objects in tabletop environments.

EVALUATION CONTEXT:
- Target scenario: Robotic manipulation, such as pick-place, push, grasp, and object relocation.
- Expected agent: robotic arm/end-effector, not human hands.
- Expected environment: tabletop with objects, typical for robot manipulation tasks.
- Expected physics: realistic robot-object interactions following physical laws.

CRITICAL EVALUATION PRINCIPLES:
1. Base all judgments only on what is visually observable in the sampled frames.
2. Do not infer information not shown.
3. Evaluate temporal coherence across the sampled frames.
4. Score only the requested dimensions.

EVALUATION DIMENSIONS AND SCORING RUBRICS:
1. Interaction_Quality: quality of robot-object interactions.
- Score 1: objects pass through robot or other objects; no proper contact.
- Score 2: contact exists but interaction is unrealistic, such as sliding without friction or incorrect force response.
- Score 3: mostly plausible interactions with minor issues, such as slight penetration or imperfect grasping.
- Score 4: realistic contact physics with proper friction, force transfer, and stable object motion.
- Score 5: excellent interaction physics; indistinguishable from real robot manipulation.

2. Perspectivity: 3D consistency and camera geometry.
- Score 1: scene has no coherent 3D structure; objects float inconsistently.
- Score 2: 3D structure is unstable, such as changing scale or incorrect occlusion.
- Score 3: reasonable 3D consistency with minor issues, such as slight perspective drift.
- Score 4: stable camera perspective with consistent depth relationships.
- Score 5: perfect camera geometry and 3D consistency.

SPECIFIC ROBOT-RELATED CHECKS:
- Robotic arm should have mechanical appearance, not human limbs.
- End-effector or gripper should maintain consistent form throughout interaction.
- Robot motion should show appropriate joint movement and kinematics.
- Object manipulation should respect object mass and inertia.
- Contact should be maintained appropriately during grasping or lifting.
"""
    if include_instruction_following:
        prompt += """
3. Instruction_Following: adherence to the given VIDEO INSTRUCTION.
- Hallucination check: if the video shows human hands instead of robotic arms, score <= 2 immediately.
- Score 1: completely different from instruction, such as wrong action, wrong objects, or wrong scene.
- Score 2: partially related but with major errors, such as wrong target object or incorrect manipulation type.
- Score 3: follows general intent but with execution errors, such as correct action sequence but imprecise outcome.
- Score 4: mostly correct with minor deviations, such as slight position error or extra unnecessary motion.
- Score 5: perfect execution of all specified elements: action, object, scene, and outcome.

OUTPUT FORMAT REQUIREMENTS:
You must output a single valid JSON object with exactly three keys:
- "Interaction_Quality"
- "Perspectivity"
- "Instruction_Following"
"""
    else:
        prompt += """
OUTPUT FORMAT REQUIREMENTS:
You must output a single valid JSON object with exactly two keys:
- "Interaction_Quality"
- "Perspectivity"
"""
    prompt += """

Each value must be an object with exactly two keys:
- "score": integer 1-5
- "reason": concise explanation citing specific visual evidence from the frames

Output only the JSON object, with no Markdown and no extra text.
Now evaluate the provided video frames based on the criteria above.
"""
    return prompt


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_qwen_vl(
    model,
    processor,
    prompt: str,
    images: Sequence[Image.Image],
    *,
    max_new_tokens: int = 400,
    temperature: float = 0.1,
) -> str:
    messages = [{
        "role": "user",
        "content": [*[{"type": "image", "image": img} for img in images], {"type": "text", "text": prompt}],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if "pixel_values" not in inputs:
        image_inputs = processor(images=list(images), return_tensors="pt")
        inputs["pixel_values"] = image_inputs.pixel_values
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    device = _model_device(model)
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
        temperature=temperature,
        early_stopping=True,
    )
    input_len = inputs["input_ids"].shape[1]
    gen_trim = generated_ids[:, input_len:]
    text = processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def safe_parse_scores(raw_text: str) -> Optional[Dict]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    candidates = [raw_text]
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_metrics(parsed: Optional[Dict], metric_keys: Sequence[str] = METRIC_KEYS) -> Dict[str, Dict]:
    metrics: Dict[str, Dict] = {}
    for key in metric_keys:
        value = parsed.get(key, {}) if isinstance(parsed, dict) else {}
        score = value.get("score", 0) if isinstance(value, dict) else 0
        try:
            score = int(round(float(score)))
        except Exception:
            score = 0
        score = max(0, min(5, score))
        reason = value.get("reason", "") if isinstance(value, dict) else ""
        metrics[key] = {
            "score": score,
            "reason": str(reason),
            "score_normalized": round(score / 5.0, 4) if score else 0.0,
        }
    return metrics


def _flat_metric_scores(metrics: Dict[str, Dict], metric_keys: Sequence[str] = METRIC_KEYS) -> Dict[str, float]:
    return {f"{key}_score": metrics[key]["score"] for key in metric_keys}


def _resolve_model_path(model_path: Optional[str], config_path: Optional[Path]) -> str:
    if model_path:
        return model_path
    if config_path and config_path.exists():
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            candidate = cfg.get("ckpt", {}).get("vlm_model")
            if candidate:
                return str(candidate)
        except Exception:
            pass
    if DEFAULT_MODEL_PATH:
        return DEFAULT_MODEL_PATH
    raise ValueError("Qwen3-VL model path is required. Pass --model_path or set QWEN3_VL_MODEL.")


def _resolve_torch_dtype(torch_dtype: str):
    if torch_dtype == "auto":
        return "auto"
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(torch_dtype, torch_dtype)


def load_qwen3_vl_model(model_path: str, *, torch_dtype: str = "auto", device_map: str = "auto", trust_remote_code: bool = False):
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "compute_qwen_vl_action_quality.py requires a transformers version with "
            "Qwen3VLForConditionalGeneration."
        ) from exc

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=_resolve_torch_dtype(torch_dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return model, processor


def compute_qwen_vl_action_quality_for_videos(
    pred_root: Path,
    true_root: Optional[Path] = None,
    *,
    instruction_json: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    model_path: Optional[str] = None,
    config_path: Optional[Path] = None,
    model_name: str = "qwen3_vl",
    num_frames: int = 16,
    skip_first: int = 0,
    start_fraction: float = 0.0,
    max_new_tokens: int = 400,
    temperature: float = 0.1,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False,
    keep_raw: bool = True,
    include_instruction_following: bool = True,
    max_samples: Optional[int] = None,
) -> Dict:
    pred_root = Path(pred_root)
    true_root = Path(true_root) if true_root else None
    output_dir = Path(output_dir) if output_dir else pred_root / "metrics"
    raw_dir = Path(raw_dir) if raw_dir else output_dir / "qwen_vl_raw"
    if keep_raw:
        ensure_dir(raw_dir)

    resolved_model_path = _resolve_model_path(model_path, config_path)
    model, processor = load_qwen3_vl_model(
        resolved_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    instruction_map = load_manifest_instruction_map(pred_root)
    instruction_map.update(load_instruction_map(instruction_json))
    videos = list_pred_videos(pred_root, true_root)
    if max_samples is not None:
        videos = videos[:max_samples]
    metric_keys = METRIC_KEYS if include_instruction_following else NO_INSTRUCTION_METRIC_KEYS

    per_sample = []
    for sample_id, video_path in tqdm(videos, desc="qwen_vl_action_quality", ncols=100):
        item = {
            "sample_id": sample_id,
            "pred_video": str(video_path),
            "instruction": instruction_map.get(sample_id, instruction_map.get(video_path.name, "")),
            "raw_response_file": None,
            "error": None,
        }
        frames = sample_video_frames(
            video_path,
            num_frames=num_frames,
            skip_first=skip_first,
            start_fraction=start_fraction,
        )
        if not frames:
            item["error"] = "no_frames"
            per_sample.append(item)
            continue

        try:
            prompt = build_action_quality_prompt(
                item["instruction"],
                include_instruction_following=include_instruction_following,
            )
            raw_text = run_qwen_vl(
                model,
                processor,
                prompt,
                frames,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if keep_raw:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_file = raw_dir / f"{video_path.stem}_{timestamp}.json"
                save_json({"raw_response": raw_text}, raw_file)
                item["raw_response_file"] = str(raw_file)

            parsed = safe_parse_scores(raw_text)
            if parsed is None:
                item["error"] = "parse_failed"
                item["raw_response"] = raw_text
                metrics = normalize_metrics(None, metric_keys)
            else:
                metrics = normalize_metrics(parsed, metric_keys)
            item["metrics"] = metrics
            item.update(_flat_metric_scores(metrics, metric_keys))
        except Exception as exc:
            item["error"] = str(exc)
        per_sample.append(item)

    result = {
        "metric": "qwen_vl_action_quality",
        "backend": "Qwen3VLForConditionalGeneration",
        "model_name": model_name,
        "model_path": resolved_model_path,
        "num_samples": len(per_sample),
        "num_frames": num_frames,
        "skip_first": skip_first,
        "start_fraction": start_fraction,
        "metric_keys": list(metric_keys),
        "include_instruction_following": include_instruction_following,
        "per_sample": sorted(per_sample, key=lambda row: row["sample_id"]),
    }
    for key in metric_keys:
        result.update(summarize_values([row.get(f"{key}_score") for row in per_sample], key))
    return result


def compute_qwen_vl_action_quality(pred_folder, true_folder=None, **kwargs):
    result = compute_qwen_vl_action_quality_for_videos(
        Path(pred_folder),
        Path(true_folder) if true_folder else None,
        **kwargs,
    )
    metric_keys = result.get("metric_keys", METRIC_KEYS)
    return {
        key: result.get(f"mean_{key}")
        for key in metric_keys
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Use Qwen3-VL to evaluate robotic action quality in generated videos.")
    parser.add_argument("--pred_root", "--video_dir", dest="pred_root", required=True)
    parser.add_argument("--true_root", default=None, help="Optional GT video folder, used only for matching sample layout.")
    parser.add_argument("--instruction_json", "--summary_json", dest="instruction_json", default=None)
    parser.add_argument("--metrics", default="all", help="Compatibility option; all Qwen3-VL metrics are always computed.")
    parser.add_argument("--output_dir", "--output_root", dest="output_dir", default=None)
    parser.add_argument("--raw_dir", "--tmp_root", dest="raw_dir", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_name", default="qwen3_vl")
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--skip_first", type=int, default=0)
    parser.add_argument("--start_fraction", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_keep_raw", action="store_true")
    parser.add_argument("--skip_instruction_following", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    output_dir = Path(args.output_dir) if args.output_dir else pred_root / "metrics"
    result = compute_qwen_vl_action_quality_for_videos(
        pred_root,
        Path(args.true_root) if args.true_root else None,
        instruction_json=Path(args.instruction_json) if args.instruction_json else None,
        output_dir=output_dir,
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        model_path=args.model_path,
        config_path=Path(args.config_path) if args.config_path else None,
        model_name=args.model_name,
        num_frames=args.num_frames,
        skip_first=args.skip_first,
        start_fraction=args.start_fraction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        keep_raw=not args.no_keep_raw,
        include_instruction_following=not args.skip_instruction_following,
        max_samples=args.max_samples,
    )
    path = save_json(result, metric_output_path(output_dir, "qwen_vl_action_quality"))
    means = " ".join(f"{key}={result.get(f'mean_{key}')}" for key in result.get("metric_keys", METRIC_KEYS))
    print(f"{means} -> {path}")


if __name__ == "__main__":
    main()
