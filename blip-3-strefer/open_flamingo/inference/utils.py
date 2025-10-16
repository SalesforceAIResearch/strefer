from omegaconf import OmegaConf
import cv2
import numpy as np
import torch
import torchvision
import sys
import os
sys.path.append('..')
    
def set_model_conf(args):
    cfg = dict(
        model_family = 'kosmos',
        lm_path = 'microsoft/Phi-3-mini-4k-instruct',
        tokenizer_path = 'microsoft/Phi-3-mini-4k-instruct',
        conv_template_name = 'phi_3',
        vision_encoder_path = 'google/siglip-so400m-patch14-384',
        vision_encoder_pretrained = 'google',
        num_vision_tokens = 128,
        image_aspect_ratio = 'pad',
        anyres_patch_sampling = False,
        repeat_latents = False,
        ckpt_pth = args.ckpt_pth,
    )
    cfg = OmegaConf.create(cfg)

    if cfg.model_family in ['kosmos-instruct', 'kosmos', 'llava']:
        additional_kwargs = {
            "image_aspect_ratio": cfg.image_aspect_ratio,
            }
    if cfg.model_family in ['kosmos-instruct', 'kosmos']:
        print(f'Using temporal encoder: {args.temporal_encoder}')
        additional_kwargs.update({
            "num_vision_tokens": cfg.num_vision_tokens,
            "repeat_latents": cfg.repeat_latents,
            "anyres_patch_sampling": cfg.anyres_patch_sampling,
            "temporal_encoder": args.temporal_encoder,
            "num_timestamp_tokens": args.num_timestamp_tokens,
            "mask_referring": args.mask_referring,
        })

    return cfg, additional_kwargs
    
def apply_prompt_template(prompt, cfg, prompt_type, subtitles=None):
    assert 'Phi-3' in cfg.lm_path
    assert prompt_type == "general"
    s = (
            '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
            f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
        )
    return s

def get_prompt(question, tokenizer, cfg, prompt_type, subtitles=None):
    prompt = "<image>\n" + question
    prompt = apply_prompt_template(prompt, cfg, prompt_type, subtitles)
    lang_x = tokenizer([prompt], return_tensors="pt")
    return prompt, lang_x
