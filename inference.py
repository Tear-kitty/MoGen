import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image

import numpy as np
import cv2
import time
from ip_adapter_XL import IPAdapterXL
from transformers import AutoImageProcessor, AutoModel
import json

PALETTE = [
    (255,255,255), (255,0,0), (0,255,0), (0,0,255),
    (255,255,0), (255,0,255), (0,255,255),
    (128,128,128), (255,128,0), (0,128,255),
    (128,0,128), (0,128,0), (128,0,0), (0,0,128),
    (128,128,0), (0,128,128), (192,192,192), (64,64,64),
    (255,64,64), (64,255,64)
]

import hashlib

def _label_to_color(label):
    if isinstance(label, int):
        return PALETTE[label % len(PALETTE)]
    h = int(hashlib.md5(str(label).encode()).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]

def create_canvas(H=448, W=448):
    return np.zeros((H, W, 3), dtype=np.uint8)

def draw_one_box(canvas, box, label=None, thickness=3, fill=False, clamp=True):
    H, W, _ = canvas.shape
    l,t,r,b = box
    if clamp:
        l = max(0, min(W-1, int(l))); r = max(0, min(W-1, int(r)))
        t = max(0, min(H-1, int(t))); b = max(0, min(H-1, int(b)))
    if r <= l or b <= t:
        return canvas
    color = _label_to_color(label) if label is not None else (255,255,255)
    if fill:
        cv2.rectangle(canvas, (l,t), (r,b), color, -1)
    cv2.rectangle(canvas, (l,t), (r,b), color, thickness)
    return canvas

def to_pil(canvas):
    return Image.fromarray(canvas)

from torchvision import transforms
transform_mask = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

def load_image(path):
    img = Image.open(path).convert("RGBA")           
    bg = Image.new("RGB", img.size, (255, 255, 255))  
    bg.paste(img, mask=img.split()[3])               
    return bg

if __name__ == "__main__":
    device = "cuda"
    base_model_path = "~/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b/"
    ip_ckpt = "~/checkpoints/checkpoint-0/text_embedding_projector.bin"
    dino_v2 = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant').to(device)
    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-giant')

    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()

    prompt = 'three pandas and one puppy are standing together in shallow water' 
    image_path = None #'~/data/image/3.png'
    box_json_path = '~/data/box/3_box.json'
    appearance_path = '~/data/object/3/'

    if image_path is not None:
        image_reference = Image.open(image_path)
        # original_width, original_height = image_reference.size
        structure_ref = dino_processor(images=image_reference, return_tensors="pt").pixel_values
        sturcture_ref_embeds = dino_v2(structure_ref.to(device)).last_hidden_state
    else:
        sturcture_ref_embeds=None

    if box_json_path is not None:
        with open(box_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        original_width =  data['imageHeight']
        original_height = data['imageWidth']
        label=0
        canvas = create_canvas(H=512, W=512)
        box_mask = []
        for shape in data['shapes']:
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, y1, x2, y2 = (int(x1*512/original_width)), (int(y1*512/original_width)), (int(x2*512/original_width)), (int(y2*512/original_width))
            if x1>512 or y1>512 or x2>512 or y2>512:
                raise ValueError('xy>512')
            box=[x1, y1, x2, y2]
            canvas = draw_one_box(canvas, box, label, thickness=3, fill=False)
            label+=1
        box_mask = to_pil(canvas)
        box_mask.save(f"box.png") 
        box_img=transform_mask(box_mask)
    else:
        box_img=None

    if appearance_path is not None:
        if os.path.exists(appearance_path):
            images = [load_image(os.path.join(appearance_path, f))
            for f in os.listdir(appearance_path) if f.lower().endswith(".png")]
            appearance_ref_tensor = [dino_processor(images=img, return_tensors="pt").pixel_values[0] for img in images]
            appearance_ref_tensor = torch.stack(appearance_ref_tensor, dim=0)
            b, c, h, w = appearance_ref_tensor.shape
            if b<6:
                appearance_ref_tensor_padded = torch.zeros(6, c, h, w, dtype=appearance_ref_tensor.dtype, device=appearance_ref_tensor.device)
                appearance_ref_tensor_padded[:b] = appearance_ref_tensor
            appearance_ref_num = torch.tensor(len(appearance_ref_tensor)).unsqueeze(0)
            appearance_ref = appearance_ref_tensor_padded.unsqueeze(0)

            b, n, c, h, w = appearance_ref.shape
            appearance_ref = appearance_ref.reshape(b*n,c,h,w).half().cuda()
            image_reference_embeds = dino_v2(appearance_ref).last_hidden_state
            image_reference_embeds = image_reference_embeds.reshape(b,n,261,1536)
            mask = torch.arange(image_reference_embeds.size(1)).unsqueeze(0).to(image_reference_embeds.device) < appearance_ref_num.unsqueeze(1).cuda()
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            image_reference_embeds = image_reference_embeds * mask
            image_reference_embeds = image_reference_embeds.reshape(b,n*261,1536)
    else:
        image_reference_embeds=None

    # generate image
    num_samples = 4

    if image_path is None and box_json_path is None and appearance_path is None:
        target_blocks = ['down_blocks.2.attentions.1']
        ip_model = IPAdapterXL(pipe, ip_ckpt, device, num_tokens=128, target_blocks=target_blocks, use_control=False)
    else:
        target_blocks = ['blocks']
        ip_model = IPAdapterXL(pipe, ip_ckpt, device, num_tokens=512, target_blocks=target_blocks, use_control=True) #"block" up_blocks.0.attentions.1 down_blocks.2.attentions.1

    prompt_i = prompt #+ ', best quality, high quality'
    print({prompt})
    # start_time = time.perf_counter()
    images = ip_model.generate(
        pil_image=sturcture_ref_embeds,
        prompt=prompt_i,
        appearance=image_reference_embeds,
        box=box_img,
        # box_mask = box_mask,
        negative_prompt= None,#"text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, bad anatomy",
        scale=0.5, #attention_control
        text_scale=0.8, #text_control
        guidance_scale=5,
        num_samples=num_samples,
        num_inference_steps=30, 
        seed=42,
        )
    # end_time = time.perf_counter()
    # print(start_time-end_time)
    for i in range(num_samples):
        images[i].save(f"~/results/{prompt}_{i}.png")

