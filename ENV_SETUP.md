# MoGen 环境配置

下面以 CUDA 12.1 + PyTorch 2.4.1 为默认组合。项目默认路径保留了原服务器路径；如果服务器 CUDA / 驱动版本不同，只需要替换 PyTorch 安装命令中的 CUDA wheel。

## 1. 创建 conda 环境

```bash
conda create -n mogen python=3.10 -y
conda activate mogen
python -m pip install --upgrade pip setuptools wheel ninja packaging
```

## 2. 安装 PyTorch

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

检查 CUDA 是否可用：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
PY
```

## 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

## 4. 安装 flash-attn

```bash
MAX_JOBS=8 pip install flash-attn==2.6.3 --no-build-isolation
```

检查：

```bash
python - <<'PY'
import flash_attn
print('flash_attn import ok')
PY
```

## 5. 配置 / 测试 accelerate

本项目已提供 `acc_configs/gpu8.yaml`，可以直接用：

```bash
accelerate test --config_file acc_configs/gpu8.yaml
```

如果只想先用单卡验证：

```bash
accelerate test --config_file acc_configs/single_gpu.yaml
```

## 6. 可选依赖

当前代码不依赖 xFormers；训练默认使用 flash-attn，推理默认使用 PyTorch SDPA。如果你仍想测试 xFormers，可以单独安装并启动时加入 `--enable_xformers_memory_efficient_attention`，否则不需要。
