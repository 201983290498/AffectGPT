# AffectGPT 训练指南 / Training Guide

## 问题修复说明 / Issue Fix Description

### 问题 / Problem
在使用最新版本的 transformers (4.49.0) 时，遇到以下导入错误：
```
ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.modeling_utils'
```

When using the latest version of transformers (4.49.0), you may encounter the following import error:
```
ImportError: cannot import name 'find_pruneable_heads_and_indices' from 'transformers.modeling_utils'
```

### 解决方案 / Solution
在 transformers 4.x 版本中，以下函数已从 `transformers.pytorch_utils` 迁移到 `transformers.modeling_utils`：
- `find_pruneable_heads_and_indices`
- `apply_chunking_to_forward`
- `prune_linear_layer`

同时，`ALL_LAYERNORM_LAYERS` 已从 `transformers.pytorch_utils` 迁移到 `transformers.trainer`。

In transformers 4.x, the following functions have been moved from `transformers.pytorch_utils` to `transformers.modeling_utils`:
- `find_pruneable_heads_and_indices`
- `apply_chunking_to_forward`
- `prune_linear_layer`

Additionally, `ALL_LAYERNORM_LAYERS` has been moved from `transformers.pytorch_utils` to `transformers.trainer`.

### 修改的文件 / Modified Files
1. `AffectGPT/my_affectgpt/models/Qformer.py` - 更新导入语句 / Updated import statements
2. `OV-MER/mPLUG-Owl/mplug_owl/modeling_mplug_owl.py` - 更新导入语句 / Updated import statements
3. `OV-MER/mPLUG-Owl/mplug_owl_video/modeling_mplug_owl.py` - 更新导入语句 / Updated import statements
4. `OV-MER/LLaMA-VID/llamavid/train/llava_trainer.py` - 更新导入语句 / Updated import statements

## 环境准备 / Environment Setup

### 1. 创建 Conda 环境 / Create Conda Environment
```bash
conda env create -f AffectGPT/environment.yml
conda activate vllm2
```

### 2. 验证环境 / Verify Environment
确保安装了正确版本的依赖：
```bash
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"
# 应该输出：transformers version: 4.49.0
```

Ensure the correct version of dependencies are installed:
```bash
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"
# Should output: transformers version: 4.49.0
```

## 数据准备 / Data Preparation

### 1. 下载数据集 / Download Datasets

#### MER-Caption+ 数据集 / MER-Caption+ Dataset
从 Hugging Face 下载：https://huggingface.co/datasets/MERChallenge/MER2025

Download from Hugging Face: https://huggingface.co/datasets/MERChallenge/MER2025

数据集结构 / Dataset structure:
```bash
dataset/mer2025-dataset/
├── video/                      # 训练视频，132,171 个样本 / Training videos, 132,171 samples
├── audio/                      # 预提取的音频 / Pre-extracted audio
├── openface_face/              # 预提取的面部文件 / Pre-extracted face files
├── subtitle_chieng.csv         # 预提取的字幕内容 / Pre-extracted subtitle content
├── track2_train_mercaptionplus.csv  # MER-Caption+ 数据集（OV 标签）/ Dataset (OV labels)
└── track3_train_mercaptionplus.csv  # MER-Caption+ 数据集（描述）/ Dataset (Description)
```

### 2. 下载预训练模型 / Download Pre-trained Models
从百度网盘或 Hugging Face 下载预训练模型：

Download pre-trained models from Baidu Pan or Hugging Face:

```bash
AffectGPT/models/
├── chinese-hubert-large/       # 音频编码器 / Audio encoder
│                              # 下载链接 / Download: https://huggingface.co/TencentGameMate/chinese-hubert-large
├── clip-vit-large-patch14/     # 视频编码器 / Video encoder
│                              # 下载链接 / Download: https://huggingface.co/openai/clip-vit-large-patch14
└── Qwen2.5-7B-Instruct/        # 大语言模型 / LLM
                               # 下载链接 / Download: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

## 训练模型 / Training the Model

### 基本训练命令 / Basic Training Command
```bash
cd AffectGPT
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml
```

### 使用多 GPU 训练 / Training with Multiple GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml
```

### 使用调试模式 / Using Debug Mode
如果需要调试，可以使用 pybug：
```bash
CUDA_VISIBLE_DEVICES=0 pybug 5678 train.py \
    --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml
```

If you need to debug, you can use pybug:
```bash
CUDA_VISIBLE_DEVICES=0 pybug 5678 train.py \
    --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml
```

### 自定义训练参数 / Custom Training Parameters
使用 `--options` 参数覆盖配置文件中的设置：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml \
    --options 'run.batch_size=8' 'run.lr=1e-5'
```

Use `--options` parameter to override settings in the config file:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml \
    --options 'run.batch_size=8' 'run.lr=1e-5'
```

## 训练配置文件 / Training Configuration Files

训练配置文件位于 `train_configs/` 目录下。主要配置文件：

Training configuration files are located in the `train_configs/` directory. Main configuration files:

- `emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml` - 在 MERCaption+ 上训练，使用预提取的面部输入 / Training on MERCaption+ with pre-extracted face input
- `mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml` - 直接处理原始帧 / Direct processing of raw frames

## 推理 / Inference

### 使用预训练权重进行推理 / Inference with Pre-trained Weights

如果不想从头训练，可以使用预训练权重：

If you don't want to train from scratch, you can use pre-trained weights:

```bash
# 下载预训练权重 / Download pre-trained weights
# https://pan.baidu.com/s/1wtKBxHQP4eCUSAVuBrOzag?pwd=27sh
# 或 Hugging Face: https://huggingface.co/MERChallenge/AffectGPT

AffectGPT/output/
└── emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz/
```

### 单个视频推理 / Single Video Inference

```bash
# 生成 OV 标签 / Generate OV labels
CUDA_VISIBLE_DEVICES=0 python inference_sample.py \
    --zeroshot \
    --video_path='demo/sample_00000000.mp4' \
    --audio_path='demo/sample_00000000.wav' \
    --subtitle="I don't know! I, I, I don't have experience in this area." \
    --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml \
    --options "inference.test_epoch=30"

# 生成情感描述 / Generate emotion description
CUDA_VISIBLE_DEVICES=0 python inference_sample.py \
    --zeroshot \
    --video_path='demo/sample_00000000.mp4' \
    --audio_path='demo/sample_00000000.wav' \
    --subtitle="I don't know! I, I, I don't have experience in this area." \
    --outside_user_message="Please infer the person's emotional state and provide your reasoning process." \
    --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml \
    --options "inference.test_epoch=30"
```

### 批量推理（MER-UniBench）/ Batch Inference (MER-UniBench)

```bash
# 提示1：生成 OV 标签 / Prompt1: Generate OV labels
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python inference_hybird_batch2.py \
    --zeroshot \
    --dataset='inferenceData' \
    --options "inference.test_epochs=30-60" "inference.skip_epoch=5" \
    --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml

# 提示2：生成情感描述 / Prompt2: Generate emotion description
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python inference_hybird.py \
    --zeroshot \
    --dataset='inferenceData' \
    --outside_user_message="Please infer the person's emotional state and provide your reasoning process." \
    --options "inference.test_epochs=30-60" "inference.skip_epoch=5" "inference.base_root=output/results-description" \
    --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml
```

## 评估 / Evaluation

### 完整评估 / Full Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

### 仅计算分数 / Score Calculation Only
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation-scoreonly.py
```

## 常见问题 / FAQ

### 1. 导入错误 / Import Error
**问题 / Problem**: `ImportError: cannot import name 'find_pruneable_heads_and_indices'`

**解决方案 / Solution**: 确保你使用的是修复后的代码版本，导入已更新为从 `transformers.modeling_utils` 导入。

Ensure you are using the fixed version of the code, where imports have been updated to import from `transformers.modeling_utils`.

### 2. 显存不足 / Out of Memory
**问题 / Problem**: CUDA out of memory error

**解决方案 / Solution**: 
- 减小批量大小 / Reduce batch size
- 使用梯度累积 / Use gradient accumulation
- 使用更少的 GPU / Use fewer GPUs
```bash
python train.py --cfg-path=... --options 'run.batch_size=4'
```

### 3. Padding 方向 / Padding Direction
**重要提示 / Important Note**: 
- 训练时使用 `padding_side='right'` / Use `padding_side='right'` during training
- 推理时使用 `padding_side='left'` / Use `padding_side='left'` during inference

Padding 的方向对最后的结果有影响！/ The direction of padding affects the final results!

## 参考资源 / References

- **论文 / Paper**: [AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding](https://arxiv.org/pdf/2501.16566)
- **Hugging Face**: https://huggingface.co/MERChallenge/AffectGPT
- **数据集 / Dataset**: https://huggingface.co/datasets/MERChallenge/MER2025
- **GitHub**: https://github.com/201983290498/AffectGPT

## 许可证 / License

本项目使用 Apache 2.0 许可证。服务是研究预览版，仅供**非商业用途**。

This project is released under the Apache 2.0 license. The service is a research preview intended for **non-commercial use ONLY**.
