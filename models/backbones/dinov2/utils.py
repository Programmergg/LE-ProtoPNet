"""
Adapted from https://github.com/facebookresearch/dinov2
"""
import torch
from typing import Any
from omegaconf import OmegaConf
from urllib.parse import urlparse
from models.backbones.dinov2.models import vision_transformer as vits
from peft import LoraConfig, get_peft_model,AdaLoraConfig, AdaLoraModel
from models.backbones.dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from models.backbones.dinov2.models.vision_transformer import DinoVisionTransformer
from functools import partial
import logging
from src.lora import LoraModel


dict_model = {
    "dinov2_vits14_reg": ["facebookresearch/dinov2", "dinov2_vits14_reg"],
    "dinov2_vitb14_reg": ["facebookresearch/dinov2", "dinov2_vitb14_reg"],
    "dinov2_vitl14_reg": ["facebookresearch/dinov2", "dinov2_vitl14_reg"],
    "dinov2_vitg14_reg": ["facebookresearch/dinov2", "dinov2_vitg14_reg"],
    'dino_vitb16':['facebookresearch/dino:main', 'dino_vitb16'],
}

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import List, Literal, Optional, Union

# from peft.config import PeftConfig
# from peft.utils import PeftType
# from ..lora.config import LoftQConfig
from torch import nn
#@dataclass

# from transformers import PreTrainedModel
# from src.config import PeftConfig
# from src import (
#     TaskType,
#     LoraConfig,
#     MoleConfig,
#     AdaMoleConfig,
#     PeftTrainer,
#     PeftModelForCausalLM,
#     PeftModel,
# )
# from src.peft_model import get_peft_model
# class PeftModelForVision(PeftModel):
#     """
#     PEFT Model for Vision Tasks (e.g., Classification or Feature Extraction)
#     """
#     def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> None:
#         super().__init__(model, peft_config, adapter_name)

#     def forward(
#         self,
#         image,
#         **kwargs,
#     ) -> torch.Tensor:
#         if image is None:
#             raise ValueError("pixel_values must be provided for Vision Transformer tasks.")
#         return self.base_model(
#             image,
#             **kwargs,
#         )

def apply_lora_to_specific_layers(model, lora_config, target_layers=[1,2,3,4,5,6,7,8,9,10,11,12]):
    for i, layer in enumerate(model.blocks):
        if (i + 1) in target_layers:  # +1 because Python indexing starts from 0
            print(f"Applying LoRA to layer {i + 1}")
            layer_with_lora = get_peft_model(layer, lora_config)
            model.blocks[i] = layer_with_lora
            # for name, module in layer_with_lora.named_modules():
            #     if hasattr(module, "lora_A"):
            #         module.lora_A = LoRADropoutWrapper_A(module.lora_A, dropout_prob=0.1)
            #     if hasattr(module, "lora_B"):
            #         module.lora_B = LoRADropoutWrapper_B(module.lora_B, dropout_prob=0.1)

        else:
            for param in layer.parameters():
                param.requires_grad = False
    
    return model

def create_model(arch: str, path_weight=None):
    """
    Create a model based on the specified architecture.
    Args:
    ----
        arch (str): The architecture name.
        path_weight (str, optional): The path to the weight file. Defaults to None.
    Returns:
    -------
        torch.nn.Module: The created model.
    """
    if arch in dict_model:
        #image_encoder = torch.hub.load(*dict_model[arch])
         #lora_model = LoraModel(image_encoder, 8, 32, 0.1, modified_modules=['attn.qkv','attn.proj'])#, 'mlp.fc1', 'mlp.fc2'])
       # lora_model.add_all_delta_to_backbone(image_encoder, lora_model.modified_modules)
        image_encoder = DinoVisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention),
                                   num_register_tokens=0,block_chunks=0)
        state_dict = torch.load('/home/WRS/SPM/pretraind model/dino_vitbase16_pretrain.pth', map_location="cpu")
        msg = image_encoder.load_state_dict(state_dict, strict=False)
        #logging.info(msg)    
   
       
    elif path_weight is not None:
        image_encoder, _ = create_local_model(arch, path_weight)
    else:
        raise NotImplementedError
    image_encoder = image_encoder.to(torch.float32)
    lora_config = LoraConfig(
    r=8,               # Rank of LoRA (hyperparameter)
    lora_alpha=32,      # Scaling factor
    lora_dropout=0.1,   # Dropout for LoRA weights
    target_modules=["qkv", "proj"]  # Define target layers for LoRA (attention layers)
    )
    # lora_config = LoraConfig(
    #         lora_rank=8,
    #         lora_alpha=32,
    #         lora_dropout=0.1,
    #         target_modules=['qkv', 'proj'],
    #         task_type='none',
    #         bias="none", # Define target layers for LoRA (attention layers)
    # )
    # adalora_config = AdaLoraConfig(
    #     r=8,  
    #     lora_alpha=32,      
    #     lora_dropout=0.1,
    #     target_r=4,  # 最终的目标秩
    #     tinit=2000,  # 初始化的 warmup 步数
    #     tfinal=20000,  # 最终的 warmup 步数
    #     deltaT=100,  # mask 刷新间隔
    #     beta1=0.9,  
    #     beta2=0.999,  
    #     total_step=40000, 
    #    # target_modules=["qkv", "proj"] ,
    #     target_modules=["fc1", "fc2"] ,
    #     init_lora_weights="gaussian",
    # )
    # adamole_config = AdaMoleConfig(            
    #         lora_rank=8,
    #         lora_alpha=32,
    #         lora_dropout= 0.1,
    #         target_modules=['qkv', 'attn.proj'],
    #         task_type='none',
    #         bias="none",
    #         num_experts=12,
    #         max_threshold=0.125,
    #     )
    #image_encoder_with_lora = PeftModelForVision(image_encoder, adamole_config)
   # image_encoder_with_lora = apply_lora_to_specific_layers(image_encoder, adalora_config)
   # return image_encoder_with_lora
    return image_encoder

def create_local_model(path_config: str, path_weight: str) -> tuple[Any, torch.dtype]:
    """
    Create a local model using the provided configuration file and weight file.
    Args:
    ----
        path_config (str): The path to the configuration file.
        path_weight (str): The path to the weight file.
    Returns:
    -------
        Tuple[Any, torch.dtype]: A tuple containing the created model and the autocast dtype.
    """
    cfg = OmegaConf.load(path_config)
    model, _ = build_model(cfg.student, only_teacher=True, img_size=cfg.crops.global_crops_size)
    load_pretrained_weights(model, path_weight, checkpoint_key="teacher")
    model.cuda()
    autocast_dtype = get_autocast_dtype(cfg)
    return model, autocast_dtype

def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](**vit_kwargs, drop_path_rate=args.drop_path_rate, drop_path_uniform=args.drop_path_uniform,)
        embed_dim = student.embed_dim
    return student, teacher, embed_dim

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        # logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)

def get_autocast_dtype(config):
    teacher_dtype_str = (config.compute_precision.teacher.backbone.mixed_precision.param_dtype)
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float