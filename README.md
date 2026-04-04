# Wan 世界模型微调项目 (DiffSynth-Studio)

这是基于 DiffSynth-Studio 开发的 Wan 视频生成模型微调代码，专门用于世界模型（World Model）训练与推理，当前仓库集成了 `diffsynth-studio-rlinf` 相关改动。

## 主要内容

本项目包含了 Wan2.2-TI2V-5B 模型的训练与推理脚本，支持多帧条件输入、动作嵌入，以及面向 RLinf / Libero 数据的训练流程。

## 代码改动说明

### 1. 训练脚本改动

核心训练入口为：

```bash
python examples/wanvideo/model_training/train_rlinf.py
```

在 [examples/wanvideo/model_training/train_rlinf.py](examples/wanvideo/model_training/train_rlinf.py) 中新增或修改了以下能力：

- 兼容加载包含 `set` 的权重文件，避免序列化加载报错。
- 支持通过 `--use_wow_checkpoint` 加载 WoW checkpoint，并覆盖 `pipe.dit` 权重。
- 新增 `context_noise_sigma`，可对上下文首帧添加高斯噪声。
- 新增 `static_video_prob`，可按概率将样本变成静态视频，并将 action 置零、末维设为 `-1`。
- 支持 `RLinfNpyDataset` 与 `LiberoRGBDDataset` 两类数据集。
- 当 Libero 数据集没有独立 validation split 时，自动对 train split 做 9:1 划分，避免训练/验证数据重叠。
- 新增 `--val_interval`、`--libero_*` 等参数，方便控制验证与数据集配置。

### 2. 训练命令示例

#### RLinfNpyDataset

```bash
bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf_lerobot.sh
```

对应参数包括：

- `--extra_inputs "input_image,action"`
- `--context_noise_sigma 0.0`
- `--static_video_prob 0.05`
- `--dataset RLinfNpyDataset`
- `--dataset_base_path /workspace/qian.ren/huggingDown/RLinf/RLinf-Wan-LIBERO-Spatial/dataset`

#### LiberoRGBDDataset

```bash
bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B_rlinf.sh
```

对应参数包括：

- `--dataset LiberoRGBDDataset`
- `--libero_dataset_name /workspace/qian.ren/huggingDown/libero_object_lerobot_mask_depth`
- `--libero_context_frames 5`
- `--libero_prediction_frames 8`
- `--libero_image_size 256 256`
- `--libero_action_dim 7`
- `--libero_max_depth 5.0`

### 3. 推理脚本

batch size = 1 的推理脚本为：

```bash
python examples/wanvideo/model_inference/Wan2.2-TI2V-5B-rlinf-bs_1.py
```

该脚本支持：

- 读取 `rgb.npy` 与 `actions.npy` 进行滑窗式长视频生成。
- 通过 `input_image` 和 `input_image4` 维护上下文帧。
- 将生成结果与 GT 帧拼接可视化，并导出为视频文件。

## 注意事项

训练时请将 `diffsynth/models/model_manager.py` 中对应配置改为 `True`（原 README 提示为第 118 和 142 行，请以当前代码实际行为为准）。
