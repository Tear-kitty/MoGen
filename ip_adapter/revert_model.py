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
    names_1 = ['down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor.to_v2_ip.weight',               
               'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor.to_v2_ip.weight',
               'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_k_ip.weight',
               'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_v1_ip.weight',
               'mid_block.attentions.0.transformer_blocks.0.attn2.processor.to_v2_ip.weight']

    names_2 = [
        "1.to_k_ip.weight", "1.to_v1_ip.weight", "1.to_v2_ip.weight", "3.to_k_ip.weight", "3.to_v1_ip.weight", "3.to_v2_ip.weight", "5.to_k_ip.weight",
        "5.to_v1_ip.weight", "5.to_v2_ip.weight", "7.to_k_ip.weight", "7.to_v1_ip.weight", "7.to_v2_ip.weight", "9.to_k_ip.weight", "9.to_v1_ip.weight",
        "9.to_v2_ip.weight", "11.to_k_ip.weight", "11.to_v1_ip.weight", "11.to_v2_ip.weight", "13.to_k_ip.weight", "13.to_v1_ip.weight", "13.to_v2_ip.weight", 
        "15.to_k_ip.weight", "15.to_v1_ip.weight", "15.to_v2_ip.weight", "17.to_k_ip.weight", "17.to_v1_ip.weight", "17.to_v2_ip.weight", "19.to_k_ip.weight", 
        "19.to_v1_ip.weight", "19.to_v2_ip.weight", "21.to_k_ip.weight", "21.to_v1_ip.weight", "21.to_v2_ip.weight", "23.to_k_ip.weight", "23.to_v1_ip.weight", 
        "23.to_v2_ip.weight", "25.to_k_ip.weight", "25.to_v1_ip.weight", "25.to_v2_ip.weight", "27.to_k_ip.weight", "27.to_v1_ip.weight", "27.to_v2_ip.weight", 
        "29.to_k_ip.weight", "29.to_v1_ip.weight", "29.to_v2_ip.weight", "31.to_k_ip.weight", "31.to_v1_ip.weight", "31.to_v2_ip.weight"
    ]

    mapping = {k: v for k, v in zip(names_1, names_2)}

    import torch
    from safetensors.torch import load_file
    sd = load_file(ckpt) #"./save_model/checkpoint-24/model.safetensors"
    image_proj_sd = {}
    ip_sd = {}
    coordinates_stepembedding = {}
    noisecheck_time_stepembedding = {}
    for k in sd:
        if k.startswith("prompt_proj_model"):
            image_proj_sd[k.replace("prompt_proj_model.", "")] = sd[k]
        elif "_ip." in k:
            ip_sd[mapping[k.replace("unet.", "")]] = sd[k]
        elif k.startswith("coordinates_stepembedding"):
            coordinates_stepembedding[k.replace("coordinates_stepembedding.", "")] = sd[k]
        elif k.startswith("noisecheck_time_stepembedding"):
            noisecheck_time_stepembedding[k.replace("noisecheck_time_stepembedding.", "")] = sd[k]

    torch.save({"image_proj": image_proj_sd, 
                "ip_adapter": ip_sd,
                "coordinates_stepembedding": coordinates_stepembedding,
                "noisecheck_time_stepembedding": noisecheck_time_stepembedding
                },
               outpdir_adapter_bin)  # 只存有image_project的参数和ipattn的转换成kv的权重。 "./save_model/ip_adapter.bin"
    print('ip_adapter.bin saved')