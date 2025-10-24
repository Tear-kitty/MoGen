# MoGen: An Adaptive Framework for Controllable Multi-Object Image Generation
The source code of MoGen.

## 1. Todo
- [x] Release training code
- [x] Release evaluation code
- [ ] Release training dataset

## 2. Text-to-image Training

To train the text-to-image model, run the following command:

```bash
python train_add_box.py --train_data_dir 'MoGen/data' \
                        --output_dir 'MoGen/checkpoints' \
                        --train_batch_size 8 \
                        --max_train_steps 10000 \
                        --learning_rate 5e-05 \
                        --train_text True
```

## 3. Adaptive Control Training

To further train the adaptive control, run the following command:
```bash
python train_add_box.py --train_data_dir 'MoGen/data/' \
                        --output_dir 'MoGen/checkpoints/' \
                        --train_batch_size 6 \
                        --max_train_steps 10000\
                        --learning_rate 5e-05\
                        --train_text False \
                        --ckpt_path "MoGen/checkpoint/text_embedding_projector.bin"
```
## 4. Inference
To inference the model, run the following command:
Please refer to the code for reasoning cases.
```bash
python inference.py 
```
