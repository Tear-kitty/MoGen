# MoGen: An Adaptive Framework for Controllable Multi-Object Image Generation
<img width="2752" height="1536" alt="Gemini_Generated_Image_hgj7iyhgj7iyhgj7" src="https://github.com/user-attachments/assets/0e404374-8bf3-4d0b-981a-5578a8f10ae9" />
The source code of MoGen.

## 📄 Paper
[![arXiv](https://img.shields.io/badge/arXiv-2601.05546-b31b1b.svg)](https://arxiv.org/abs/2601.05546)

## 🎨 Qualitative Performance
<img width="1081" height="1194" alt="e3bae49a-95bd-4964-b588-30e217a0b495" src="https://github.com/user-attachments/assets/0c8657b2-6453-4861-b8a4-6c103a967c5a" />

## 1. Todo
- [x] Release training code
- [x] Release evaluation code
- [ ] Release training dataset

## 2. Text-to-image Training

A small subset of our dataset to start the training: https://drive.google.com/drive/folders/1of2PCY_9rTvj9CQX5AvsRAvq_Yg9rzqn?usp=sharing

To train the text-to-image model, run the following command:

```bash
python train_add_box.py --train_data_dir 'MoGen/data' \
                        --output_dir 'MoGen/checkpoints' \
                        --train_batch_size 8 \
                        --max_train_steps 20000 \
                        --learning_rate 5e-05 \
                        --train_text True
```

## 3. Adaptive Control Training

To further train the adaptive control, run the following command:
```bash
python train_add_box.py --train_data_dir 'MoGen/data/' \
                        --output_dir 'MoGen/checkpoints/' \
                        --train_batch_size 6 \
                        --max_train_steps 20000\
                        --learning_rate 5e-05\
                        --train_text False \
                        --ckpt_path "MoGen/checkpoint/text_embedding_projector.bin"
```
## 4. Inference
To inference the model, run the following command:
```bash
python inference.py 
```
Before launch the inference.py, please set the text prompt, structure reference, box reference or object reference:
```bash
prompt = '~' text prompt 
image_path = None or '~.png' structure reference
box_json_path = None or '~.json' box reference
appearance_path = None or '~.png' object reference
```
If 'None', it means that the current control signal is not used. Box reference and object reference are not necessary simultaneously.
