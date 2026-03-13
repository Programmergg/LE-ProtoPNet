# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
import os
import logging
import warnings
from torch import nn
from torch import Tensor
import torch

from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import math 
from loralib.loralib.adalora import SVDLinear
import torch.nn.functional as F

logger = logging.getLogger("dinov2")

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind
        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class LoraExpert(nn.Module):
    """
    LoRA Expert
    """

    def __init__(self, lora_A: nn.Module, lora_B: nn.Module, lora_dropout: nn.Module, scaling: float):
        super().__init__()
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_dropout = lora_dropout
        self.scaling = scaling
        self.dropout_prob = 0.1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        outputs = self.lora_B(self.lora_A(self.lora_dropout(inputs))) * self.scaling
        return outputs

class AdaMoeLayer(nn.Module):
    """
    Adaptive Mixture of Experts (MoE) Layer
    """
    def __init__(self, experts: nn.ModuleList, gate: nn.Module, threshold_fn: nn.Module, max_threshold: float):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.threshold_fn = threshold_fn
        self.max_threshold = max_threshold
        self.layer_loss = None

    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.sum(selected_experts, dim=0)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss

    def forward(self, inputs: torch.Tensor,proto_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        gate_logits = F.softmax(self.gate(flattened_inputs), dim=-1)  #gating function W_g(x)

        similiarity = torch.mm(gate_logits,F.normalize(proto_vector,dim=-1).T)
        attn_weights = F.softmax(similiarity,dim=-1)
        weighted_inputs = torch.mm(attn_weights,F.normalize(proto_vector,dim=-1))
        weighted_inputs = F.softmax(weighted_inputs,dim=-1)
        gate_logits = gate_logits + weighted_inputs

        thresholds = F.sigmoid(self.threshold_fn(flattened_inputs)) * self.max_threshold  #threshold function W_t(x)，阈值是学出来的，但是有上界
        adapted_gate_logits = gate_logits - thresholds
        selected_experts = torch.ge(adapted_gate_logits, 0).to(torch.float)  #大于阈值的专家置1，否则置0
        weights = adapted_gate_logits * selected_experts #大于0的权重   这里可以加dropout
        weight_sums = torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
        weight_sums = torch.where(weight_sums == 0, torch.ones_like(weight_sums), weight_sums)
        weights = weights / weight_sums
        results = torch.zeros_like(self.experts[0](flattened_inputs))

        for i, expert in enumerate(self.experts):
            batch_idx = torch.where(selected_experts[:, i])[0]
            if len(batch_idx) > 0:
                results[batch_idx] += weights[batch_idx, i, None] * expert(flattened_inputs[batch_idx])

        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        if inputs.requires_grad:
            self.layer_loss = self.get_layer_loss(gate_logits=adapted_gate_logits, selected_experts=selected_experts)
        return results


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, proj_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5 
      
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
      
    def forward(self, x: Tensor, return_attn=False, 
                ) -> Tensor:
        B, N, C = x.shape      
        qkv = self.qkv(x)
        qkv = (qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))                                                                                                                                                                                                                                                                                                                                                                     
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
       
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # (B, num_heads, N, N)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
       
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return attn
        return x
  


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x, return_attn)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
    
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        if return_attn:
            attn = x.permute(0, 2, 1, 3) @ v.permute(0, 2, 3, 1)
            return attn
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    