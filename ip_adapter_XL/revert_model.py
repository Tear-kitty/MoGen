# def revert_model(ckpt, outpdir_adapter_bin):
#     names_1 = ['down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight',
#                'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
#                'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight']

#     names_2 = [
#         "1.to_k_ip.weight", "1.to_v_ip.weight", "3.to_k_ip.weight", "3.to_v_ip.weight", "5.to_k_ip.weight",
#         "5.to_v_ip.weight", "7.to_k_ip.weight", "7.to_v_ip.weight", "9.to_k_ip.weight", "9.to_v_ip.weight",
#         "11.to_k_ip.weight", "11.to_v_ip.weight", "13.to_k_ip.weight", "13.to_v_ip.weight", "15.to_k_ip.weight",
#         "15.to_v_ip.weight", "17.to_k_ip.weight", "17.to_v_ip.weight", "19.to_k_ip.weight", "19.to_v_ip.weight",
#         "21.to_k_ip.weight", "21.to_v_ip.weight", "23.to_k_ip.weight", "23.to_v_ip.weight", "25.to_k_ip.weight",
#         "25.to_v_ip.weight", "27.to_k_ip.weight", "27.to_v_ip.weight", "29.to_k_ip.weight", "29.to_v_ip.weight",
#         "31.to_k_ip.weight", "31.to_v_ip.weight"
#     ]

#     mapping = {k: v for k, v in zip(names_1, names_2)}

#     import torch
#     from safetensors.torch import load_file
#     sd = load_file(ckpt) #"./save_model/checkpoint-24/model.safetensors"
#     image_proj_sd = {}
#     ip_sd = {}
#     for k in sd:
#         if k.startswith("prompt_proj_model"):
#             image_proj_sd[k.replace("prompt_proj_model.", "")] = sd[k]
#         elif "_ip." in k:
#             ip_sd[mapping[k.replace("unet.", "")]] = sd[k]

#     torch.save({"image_proj": image_proj_sd, 
#                 #"ip_adapter": ip_sd
#                 },
#                outpdir_adapter_bin)  # 只存有image_project的参数和ipattn的转换成kv的权重。 "./save_model/ip_adapter.bin"
#     print('ip_adapter.bin saved')

def revert_model(ckpt, outpdir_adapter_bin):

    names_1 = [
    'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',
    
    'down_blocks.1.attentions.0.transformer_blocks.1.attn2.processor.to_k_ip.weight',
    'down_blocks.1.attentions.0.transformer_blocks.1.attn2.processor.to_v_ip.weight',
    
    'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight',

    'down_blocks.1.attentions.1.transformer_blocks.1.attn2.processor.to_k_ip.weight',  
    'down_blocks.1.attentions.1.transformer_blocks.1.attn2.processor.to_v_ip.weight', 

    'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',  
    'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight',

    'down_blocks.2.attentions.0.transformer_blocks.1.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.0.transformer_blocks.1.attn2.processor.to_v_ip.weight',

    'down_blocks.2.attentions.0.transformer_blocks.2.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.0.transformer_blocks.2.attn2.processor.to_v_ip.weight', 

    'down_blocks.2.attentions.0.transformer_blocks.3.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.0.transformer_blocks.3.attn2.processor.to_v_ip.weight',
    
    'down_blocks.2.attentions.0.transformer_blocks.4.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.0.transformer_blocks.4.attn2.processor.to_v_ip.weight', 

    'down_blocks.2.attentions.0.transformer_blocks.5.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.0.transformer_blocks.5.attn2.processor.to_v_ip.weight', 

    'down_blocks.2.attentions.0.transformer_blocks.6.attn2.processor.to_k_ip.weight',  
    'down_blocks.2.attentions.0.transformer_blocks.6.attn2.processor.to_v_ip.weight', 

    'down_blocks.2.attentions.0.transformer_blocks.7.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.0.transformer_blocks.7.attn2.processor.to_v_ip.weight',

    'down_blocks.2.attentions.0.transformer_blocks.8.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.0.transformer_blocks.8.attn2.processor.to_v_ip.weight',

    'down_blocks.2.attentions.0.transformer_blocks.9.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.0.transformer_blocks.9.attn2.processor.to_v_ip.weight', 

    'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',  
    'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip_2.weight',  
    # 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.1.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.1.transformer_blocks.1.attn2.processor.to_v_ip.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.processor.to_k_ip_2.weight',  
    # 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.2.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.1.transformer_blocks.2.attn2.processor.to_v_ip.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.processor.to_k_ip_2.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.3.attn2.processor.to_k_ip.weight',  
    'down_blocks.2.attentions.1.transformer_blocks.3.attn2.processor.to_v_ip.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.processor.to_k_ip_2.weight',  
    # 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.4.attn2.processor.to_k_ip.weight',  
    'down_blocks.2.attentions.1.transformer_blocks.4.attn2.processor.to_v_ip.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.processor.to_k_ip_2.weight',  
    # 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.5.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.1.transformer_blocks.5.attn2.processor.to_v_ip.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.processor.to_k_ip_2.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.6.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.1.transformer_blocks.6.attn2.processor.to_v_ip.weight',
    # 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.processor.to_k_ip_2.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.7.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.1.transformer_blocks.7.attn2.processor.to_v_ip.weight',
    # 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.processor.to_k_ip_2.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.8.attn2.processor.to_k_ip.weight', 
    'down_blocks.2.attentions.1.transformer_blocks.8.attn2.processor.to_v_ip.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.processor.to_k_ip_2.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.processor.to_v_ip_2.weight',

    'down_blocks.2.attentions.1.transformer_blocks.9.attn2.processor.to_k_ip.weight',  
    'down_blocks.2.attentions.1.transformer_blocks.9.attn2.processor.to_v_ip.weight', 
    # 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.processor.to_k_ip_2.weight',  
    # 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.processor.to_v_ip_2.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.1.attn2.processor.to_k_ip.weight',  
    'up_blocks.0.attentions.0.transformer_blocks.1.attn2.processor.to_v_ip.weight',  

    'up_blocks.0.attentions.0.transformer_blocks.2.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.2.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.3.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.3.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.4.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.4.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.5.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.5.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.6.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.6.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.7.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.7.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.8.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.8.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.0.transformer_blocks.9.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.0.transformer_blocks.9.attn2.processor.to_v_ip.weight',

    'up_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.1.transformer_blocks.4.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.4.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.1.transformer_blocks.5.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.5.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.1.transformer_blocks.6.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.6.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.1.transformer_blocks.7.attn2.processor.to_k_ip.weight',  
    'up_blocks.0.attentions.1.transformer_blocks.7.attn2.processor.to_v_ip.weight',  

    'up_blocks.0.attentions.1.transformer_blocks.8.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.8.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.1.transformer_blocks.9.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.1.transformer_blocks.9.attn2.processor.to_v_ip.weight',

    'up_blocks.0.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.2.transformer_blocks.1.attn2.processor.to_k_ip.weight',  
    'up_blocks.0.attentions.2.transformer_blocks.1.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.2.transformer_blocks.2.attn2.processor.to_k_ip.weight',  
    'up_blocks.0.attentions.2.transformer_blocks.2.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.2.transformer_blocks.3.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.2.transformer_blocks.3.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.2.transformer_blocks.4.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.2.transformer_blocks.4.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.2.transformer_blocks.5.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.2.transformer_blocks.5.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.2.transformer_blocks.6.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.2.transformer_blocks.6.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.2.transformer_blocks.7.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.2.transformer_blocks.7.attn2.processor.to_v_ip.weight', 

    'up_blocks.0.attentions.2.transformer_blocks.8.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.2.transformer_blocks.8.attn2.processor.to_v_ip.weight',

    'up_blocks.0.attentions.2.transformer_blocks.9.attn2.processor.to_k_ip.weight', 
    'up_blocks.0.attentions.2.transformer_blocks.9.attn2.processor.to_v_ip.weight', 

    'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight', 

    'up_blocks.1.attentions.0.transformer_blocks.1.attn2.processor.to_k_ip.weight', 
    'up_blocks.1.attentions.0.transformer_blocks.1.attn2.processor.to_v_ip.weight', 

    'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
    'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v_ip.weight', 

    'up_blocks.1.attentions.1.transformer_blocks.1.attn2.processor.to_k_ip.weight', 
    'up_blocks.1.attentions.1.transformer_blocks.1.attn2.processor.to_v_ip.weight', 

    'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
    'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_v_ip.weight', 

    'up_blocks.1.attentions.2.transformer_blocks.1.attn2.processor.to_k_ip.weight', 
    'up_blocks.1.attentions.2.transformer_blocks.1.attn2.processor.to_v_ip.weight', 

    'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight', 
    'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_v_ip.weight', 

    'mid_block.attentions.0.transformer_blocks.1.attn2.processor.to_k_ip.weight', 
    'mid_block.attentions.0.transformer_blocks.1.attn2.processor.to_v_ip.weight', 

    'mid_block.attentions.0.transformer_blocks.2.attn2.processor.to_k_ip.weight',  
    'mid_block.attentions.0.transformer_blocks.2.attn2.processor.to_v_ip.weight', 

    'mid_block.attentions.0.transformer_blocks.3.attn2.processor.to_k_ip.weight',  
    'mid_block.attentions.0.transformer_blocks.3.attn2.processor.to_v_ip.weight',  

    'mid_block.attentions.0.transformer_blocks.4.attn2.processor.to_k_ip.weight',  
    'mid_block.attentions.0.transformer_blocks.4.attn2.processor.to_v_ip.weight',  

    'mid_block.attentions.0.transformer_blocks.5.attn2.processor.to_k_ip.weight',  
    'mid_block.attentions.0.transformer_blocks.5.attn2.processor.to_v_ip.weight',  

    'mid_block.attentions.0.transformer_blocks.6.attn2.processor.to_k_ip.weight',  
    'mid_block.attentions.0.transformer_blocks.6.attn2.processor.to_v_ip.weight',  

    'mid_block.attentions.0.transformer_blocks.7.attn2.processor.to_k_ip.weight', 
    'mid_block.attentions.0.transformer_blocks.7.attn2.processor.to_v_ip.weight', 

    'mid_block.attentions.0.transformer_blocks.8.attn2.processor.to_k_ip.weight', 
    'mid_block.attentions.0.transformer_blocks.8.attn2.processor.to_v_ip.weight', 

    'mid_block.attentions.0.transformer_blocks.9.attn2.processor.to_k_ip.weight',
    'mid_block.attentions.0.transformer_blocks.9.attn2.processor.to_v_ip.weight',
    ]

    names_2 = [
        "1.to_k_ip.weight", "1.to_v_ip.weight",
        "3.to_k_ip.weight", "3.to_v_ip.weight",
        "5.to_k_ip.weight", "5.to_v_ip.weight", 
        "7.to_k_ip.weight", "7.to_v_ip.weight",
        "9.to_k_ip.weight", "9.to_v_ip.weight",
        "11.to_k_ip.weight", "11.to_v_ip.weight", 
        "13.to_k_ip.weight", "13.to_v_ip.weight", 
        "15.to_k_ip.weight", "15.to_v_ip.weight", 
        "17.to_k_ip.weight", "17.to_v_ip.weight", 
        "19.to_k_ip.weight", "19.to_v_ip.weight", 
        "21.to_k_ip.weight", "21.to_v_ip.weight", 
        "23.to_k_ip.weight", "23.to_v_ip.weight", 
        "25.to_k_ip.weight", "25.to_v_ip.weight", 
        "27.to_k_ip.weight", "27.to_v_ip.weight", 
        "29.to_k_ip.weight", "29.to_v_ip.weight", #"29.to_k_ip_2.weight", "29.to_v_ip_2.weight",
        "31.to_k_ip.weight", "31.to_v_ip.weight", #"31.to_k_ip_2.weight", "31.to_v_ip_2.weight",
        "33.to_k_ip.weight", "33.to_v_ip.weight", #"33.to_k_ip_2.weight", "33.to_v_ip_2.weight",
        "35.to_k_ip.weight", "35.to_v_ip.weight", #"35.to_k_ip_2.weight", "35.to_v_ip_2.weight",
        "37.to_k_ip.weight", "37.to_v_ip.weight", #"37.to_k_ip_2.weight", "37.to_v_ip_2.weight",
        "39.to_k_ip.weight", "39.to_v_ip.weight", #"39.to_k_ip_2.weight", "39.to_v_ip_2.weight",
        "41.to_k_ip.weight", "41.to_v_ip.weight", #"41.to_k_ip_2.weight", "41.to_v_ip_2.weight",
        "43.to_k_ip.weight", "43.to_v_ip.weight", #"43.to_k_ip_2.weight", "43.to_v_ip_2.weight",
        "45.to_k_ip.weight", "45.to_v_ip.weight", #"45.to_k_ip_2.weight", "45.to_v_ip_2.weight",
        "47.to_k_ip.weight", "47.to_v_ip.weight", #"47.to_k_ip_2.weight", "47.to_v_ip_2.weight",
        "49.to_k_ip.weight", "49.to_v_ip.weight", 
        "51.to_k_ip.weight", "51.to_v_ip.weight", 
        "53.to_k_ip.weight", "53.to_v_ip.weight", 
        "55.to_k_ip.weight", "55.to_v_ip.weight", 
        "57.to_k_ip.weight", "57.to_v_ip.weight", 
        "59.to_k_ip.weight", "59.to_v_ip.weight", 
        "61.to_k_ip.weight", "61.to_v_ip.weight", 
        "63.to_k_ip.weight", "63.to_v_ip.weight", 
        "65.to_k_ip.weight", "65.to_v_ip.weight", 
        "67.to_k_ip.weight", "67.to_v_ip.weight", 
        "69.to_k_ip.weight", "69.to_v_ip.weight", 
        "71.to_k_ip.weight", "71.to_v_ip.weight", 
        "73.to_k_ip.weight", "73.to_v_ip.weight", 
        "75.to_k_ip.weight", "75.to_v_ip.weight", 
        "77.to_k_ip.weight", "77.to_v_ip.weight", 
        "79.to_k_ip.weight", "79.to_v_ip.weight", 
        "81.to_k_ip.weight", "81.to_v_ip.weight", 
        "83.to_k_ip.weight", "83.to_v_ip.weight", 
        "85.to_k_ip.weight", "85.to_v_ip.weight", 
        "87.to_k_ip.weight", "87.to_v_ip.weight", 
        "89.to_k_ip.weight", "89.to_v_ip.weight", 
        "91.to_k_ip.weight", "91.to_v_ip.weight", 
        "93.to_k_ip.weight", "93.to_v_ip.weight", 
        "95.to_k_ip.weight", "95.to_v_ip.weight", 
        "97.to_k_ip.weight", "97.to_v_ip.weight", 
        "99.to_k_ip.weight", "99.to_v_ip.weight", 
        "101.to_k_ip.weight", "101.to_v_ip.weight", 
        "103.to_k_ip.weight", "103.to_v_ip.weight", 
        "105.to_k_ip.weight", "105.to_v_ip.weight", 
        "107.to_k_ip.weight", "107.to_v_ip.weight", 
        "109.to_k_ip.weight", "109.to_v_ip.weight", 
        "111.to_k_ip.weight", "111.to_v_ip.weight", 
        "113.to_k_ip.weight", "113.to_v_ip.weight", 
        "115.to_k_ip.weight", "115.to_v_ip.weight", 
        "117.to_k_ip.weight", "117.to_v_ip.weight", 
        "119.to_k_ip.weight", "119.to_v_ip.weight", 
        "121.to_k_ip.weight", "121.to_v_ip.weight", 
        "123.to_k_ip.weight", "123.to_v_ip.weight", 
        "125.to_k_ip.weight", "125.to_v_ip.weight", 
        "127.to_k_ip.weight", "127.to_v_ip.weight", 
        "129.to_k_ip.weight", "129.to_v_ip.weight", 
        "131.to_k_ip.weight", "131.to_v_ip.weight", 
        "133.to_k_ip.weight", "133.to_v_ip.weight", 
        "135.to_k_ip.weight", "135.to_v_ip.weight", 
        "137.to_k_ip.weight", "137.to_v_ip.weight",      
        "139.to_k_ip.weight", "139.to_v_ip.weight", 
    ]

    mapping = {k: v for k, v in zip(names_1, names_2)}

    import torch
    from safetensors.torch import load_file
    sd = load_file(ckpt) 
    image_adjust_sd = {}
    ip_sd = {}
    for k in sd:
        if k.startswith("text_embedding_projector"):
            image_adjust_sd[k.replace("text_embedding_projector.", "")] = sd[k]
        elif "_ip" in k:
             ip_sd[mapping[k.replace("unet.", "")]] = sd[k]

    torch.save({"text_embedding_projector": image_adjust_sd, 
                "ip_adapter": ip_sd
                },
               outpdir_adapter_bin)   
    print('text_embedding_projector.bin saved')
