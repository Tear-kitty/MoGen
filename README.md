# MoGen Refactored

这是对原 SDXL 版 MoGen / IP-Adapter 改进代码的整理版。模型结构保持原设计：文本阶段输出 64 个文本条件 token；控制阶段输出 256 个控制条件 token，并支持 structure、box、appearance 三类控制信号的任意子集。

## 主要改动

1. 代码入口整理为 `train_text.py`、`train_control.py`、`infer.py`，核心包从旧的 `ip_adapter_XL` 重命名为 `mogen`。
2. 删除硬编码 `CUDA_VISIBLE_DEVICES`，训练完全交给 `accelerate launch --config_file ...` 管理。
3. 训练 checkpoint 直接保存为 `mogen_adapter.bin`，不再通过 `revert_model.py` 二次转换。
4. 文本训练和控制训练分开启动：
   - text stage：只使用 text prompt。
   - control stage：使用 text prompt + 可用控制信号，并可用 `--init_text_checkpoint` 从第一阶段初始化。
5. 推理也分为 text / control 两种模式：control 模式可只传 structure、只传 box、只传 appearance，或传任意组合。
6. 数据增广已加入：box 小幅 jitter、structure/appearance 全局质量退化与局部扭曲、appearance 数量邻近错误注入。
7. CFG 训练与推理做了对齐：训练阶段有文本/控制 dropout；推理负分支走空文本与空控制分支，而不是无条件复用正分支。
8. projection attention 训练默认 `flash`，推理默认 `sdpa`；如要推理也强制 flash，可传 `--attention_backend flash`。

## 环境配置

详见 [ENV_SETUP.md](ENV_SETUP.md)。最短路径如下：

```bash
conda create -n mogen python=3.10 -y
conda activate mogen
python -m pip install --upgrade pip setuptools wheel ninja packaging
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
MAX_JOBS=8 pip install flash-attn==2.6.3 --no-build-isolation
accelerate test --config_file acc_configs/gpu8.yaml
```

## 数据目录约定

默认保留原服务器路径：

```text
/home/lyf/2D/MoEdit-pami-github/data_test/
├── image/       # 训练目标图像，如 0.png
├── text/        # 文本 json，如 0.json
├── box/         # labelme box json，可缺省
├── structure/   # structure 参考图，可缺省
└── object/      # appearance 参考，如 object/47/five_xxx.png，可缺省
```

如果实际数据在其他位置，启动时传 `--train_data_dir /your/path` 即可。

## 多卡训练

### 第一阶段：text-only

```bash
accelerate launch --config_file acc_configs/gpu8.yaml train_text.py \
  --train_data_dir /home/lyf/2D/MoEdit-pami-github/data_test/ \
  --output_dir /home/lyf/2D/MoEdit-pami-github/checkpoints/ \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_train_steps 100000 \
  --save_steps 1000 \
  --mixed_precision fp16 \
  --text_drop_prob 0.10 \
  --attention_backend flash
```

训练完成后默认保存：

```text
/home/lyf/2D/MoEdit-pami-github/checkpoints/text-final/mogen_adapter.bin
```

### 第二阶段：control

```bash
accelerate launch --config_file acc_configs/gpu8.yaml train_control.py \
  --train_data_dir /home/lyf/2D/MoEdit-pami-github/data_test/ \
  --output_dir /home/lyf/2D/MoEdit-pami-github/checkpoints/ \
  --init_text_checkpoint /home/lyf/2D/MoEdit-pami-github/checkpoints/text-final/mogen_adapter.bin \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_train_steps 100000 \
  --save_steps 1000 \
  --mixed_precision fp16 \
  --attention_backend flash \
  --box_jitter_prob 0.50 \
  --structure_degrade_prob 0.35 \
  --appearance_degrade_prob 0.45 \
  --appearance_count_error_prob 0.20
```

训练完成后默认保存：

```text
/home/lyf/2D/MoEdit-pami-github/checkpoints/control-final/mogen_adapter.bin
```

### 断点恢复

```bash
accelerate launch --config_file acc_configs/gpu8.yaml train_control.py \
  --resume_from_checkpoint latest \
  --output_dir /home/lyf/2D/MoEdit-pami-github/checkpoints/ \
  ...其他参数同上
```

## 推理

### 只用 text prompt

```bash
python infer.py \
  --mode text \
  --checkpoint /home/lyf/2D/MoEdit-pami-github/checkpoints/text-final/mogen_adapter.bin \
  --prompt "six reddish-brown mushrooms, in the misty forest" \
  --num_samples 4 \
  --guidance_scale 5 \
  --num_inference_steps 30 \
  --seed 43 \
  --attention_backend sdpa
```

### text + 任意控制信号

下面示例同时使用三种控制；只想用其中一种时，删掉其他参数即可。

```bash
python infer.py \
  --mode control \
  --checkpoint /home/lyf/2D/MoEdit-pami-github/checkpoints/control-final/mogen_adapter.bin \
  --prompt "six reddish-brown mushrooms, in the misty forest" \
  --structure_image /home/lyf/2D/MoEdit-pami-github/data/structure/45.png \
  --box_json /home/lyf/2D/MoEdit-pami-github/data_test/box/62.json \
  --appearance_dir /home/lyf/2D/MoEdit-pami-github/data/object/47/ \
  --num_samples 4 \
  --guidance_scale 5 \
  --num_inference_steps 30 \
  --seed 43 \
  --attention_backend sdpa
```

输出默认写入：

```text
/home/lyf/2D/MoEdit-pami-github/results/
```

## 数据增广默认比例

- box jitter：`0.50`。只做小幅平移/缩放/长宽比扰动，并要求新旧 box IoU ≥ `0.75`。
- structure 退化：`0.35`。低分辨率 resize、模糊、JPEG artifact、轻微噪声、局部 warp 混合。
- appearance 退化：`0.45`。appearance 更容易过拟合，所以默认略高。
- appearance 数量错误：`0.20`。80% 保持原数量，20% 在邻近数量中随机选一个，例如 five 可能变成 four/six，少量 three/seven。
- CFG dropout：text stage 默认文本 drop `0.10`；control stage 默认 text-only drop `0.05`、control-only drop `0.05`、both drop `0.05`。

这些比例是偏稳的起点；如果控制信号过强、模型不听文本，可以提高 text/control dropout；如果控制跟随不足，可以降低对应退化比例或降低 dropout。

## 注意事项

- 代码里不再设置 `CUDA_VISIBLE_DEVICES`。如果要指定 GPU，在命令前加环境变量，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file acc_configs/gpu8.yaml train_text.py ...
```

这时需要把 `acc_configs/gpu8.yaml` 的 `num_processes` 改成实际 GPU 数。

- 旧 checkpoint 读取兼容 `text_embedding_projector` / `ip_adapter` 键；新 checkpoint 使用 `mogen_projector` / `mogen_attention` 键。
- control 推理必须至少提供一个控制信号；没有控制信号时请使用 text 模式和 text-stage checkpoint。
