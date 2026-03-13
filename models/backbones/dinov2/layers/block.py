# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import os
import torch
import logging
import warnings
from torch import nn, Tensor
from .attention import Attention, MemEffAttention, AdaMoeLayer, LoraExpert
from typing import Callable, List, Any, Tuple, Dict

from .mlp import Mlp
from .drop_path import DropPath
from .layer_scale import LayerScale
import torch.nn.functional as F
from torch.nn import  Dropout, LayerNorm
import math


logger = logging.getLogger("dinov2")

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import fmha, scaled_index_add, index_select_cat
        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Block)")

class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.q_scale = nn.Parameter(torch.ones(1,self.linear_a_q.out_features))
        self.v_scale = nn.Parameter(torch.ones(1,self.linear_a_v.out_features))

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv
    
class _LoRA_proj(nn.Module):
    def __init__(
            self,
            proj: nn.Module,
            linear_a: nn.Module,
            linear_b: nn.Module,
    ):
        super().__init__()
        self.proj = proj
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.dim = proj.in_features

    def forward(self, x):
        proj = self.proj(x)  # B,N,org_C
        new_proj = self.linear_b(self.linear_a(x))
        proj += new_proj
        return proj 

class arc_LoRA_attn(nn.Module):
    def __init__(
            self,
            qkv,
            w_a_q,
            w_b_q,
            w_a_v,
            w_b_v,
    ):
        super().__init__()
        self.qkv = qkv
        self.w_a_q = w_a_q
        self.w_b_q = w_b_q
        self.w_a_v = w_a_v
        self.w_b_v = w_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.q_scale = nn.Parameter(torch.ones(1,self.w_a_q.shape[1]))
        self.v_scale = nn.Parameter(torch.ones(1,self.w_a_v.shape[1]))

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = torch.matmul(x, self.w_a_q*self.q_scale,self.w_b_q)
        new_v = torch.matmul(x, self.w_a_v*self.v_scale,self.w_b_v)
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv
    
class arc_LoRA_proj(nn.Module):
    def __init__(
            self,
            proj,
            w_a,
            w_b,
    ):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b
        self.dim = proj.in_features
        self.scale = nn.Parameter(torch.ones(1,self.w_a.shape[1]))
    def forward(self, x):
        proj = self.proj(x)  # B,N,org_C
        new_proj = torch.matmul(x, self.w_a*self.scale, self.w_b)
        proj += new_proj
        return proj 
    
class ARC_adapter(nn.Module):
    def __init__(self, adapter_dim, hidden_dim, dropout=0.0, position='att'):
        super(ARC_adapter, self).__init__()
        self.adapter_rescale = nn.Parameter(torch.empty(1, adapter_dim))
        self.adapter_bias = nn.Parameter(torch.empty(hidden_dim))
        self.dropout = Dropout(dropout)

        if position == 'att':
            nn.init.zeros_(self.adapter_rescale)
        else:
            nn.init.xavier_uniform_(self.adapter_rescale)
        nn.init.zeros_(self.adapter_bias)

    def forward(self, x, down_projection, up_projection):
        adapter_output = torch.matmul(x, down_projection * self.adapter_rescale)
        adapter_output = self.dropout(adapter_output)
        adapter_output = torch.matmul(adapter_output, up_projection) + self.adapter_bias
        output = adapter_output + x

        return output

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        identity = x
        x = self.projector(x)

        return identity + x

class Mona(nn.Module):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOp(64)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        project1 = self.project1(x)
        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

class EVA(nn.Module):
    def __init__(self, m, n, alpha):
        super(EVA, self).__init__()
        self.s1_d = nn.Parameter(torch.randn(m, alpha))  # s1 ∈ R^(m×α)
        self.t1_d = nn.Parameter(torch.randn(alpha, n))  # t1 ∈ R^(α×n)
        self.s2_d = nn.Parameter(torch.randn(m, alpha))  # t2 ∈ R^(α×n)
        self.t2_d = nn.Parameter(torch.randn(alpha, n)) 

        self.s1_u = nn.Parameter(torch.randn(n, alpha))  # s1 ∈ R^(m×α)
        self.t1_u = nn.Parameter(torch.randn(alpha, m))  
        self.s2_u = nn.Parameter(torch.randn(n, alpha))  # s2 ∈ R^(m×α)
        self.t2_u = nn.Parameter(torch.randn(alpha, m))  # t2 ∈ R^(α×n)

    def forward(self, x):
        residual = x
        downsample = torch.mm(self.s1, self.t1) + torch.mm(self.s2, self.t2)
        x = torch.mm(downsample, x)
        upsample = torch.mm(self.s2, self.t2) + torch.mm(self.s1, self.t1 )
        x = torch.mm(upsample, x)
        output = residual + x
        
        return output

class multilora(nn.Module):
    def __init__(self, experts: nn.ModuleList, gate: nn.Module, threshold_fn: nn.Module, max_threshold: float):
        super().__init__()
        self.experts = experts
        self.gate = gate
        self.threshold_fn = threshold_fn
        self.max_threshold = max_threshold
        self.layer_loss = None  # s1 ∈ R^(m×α)
    def forward(self, inputs: torch.Tensor,proto_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        results = torch.zeros_like(self.experts[0](flattened_inputs))
        
        for i, expert in enumerate(self.experts):
            results += expert(flattened_inputs)

        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        # if inputs.requires_grad:
        #     self.layer_loss = self.get_layer_loss(gate_logits=adapted_gate_logits, selected_experts=selected_experts)
        return results

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, proj_bias: bool = True, ffn_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0, init_values=None, drop_path: float = 0.0, act_layer: Callable[..., nn.Module] = nn.GELU, norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
                 attn_class: Callable[..., nn.Module] = Attention, ffn_layer: Callable[..., nn.Module] = Mlp) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = (LayerScale(dim, init_values=init_values) if init_values else nn.Identity())
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = (LayerScale(dim, init_values=init_values) if init_values else nn.Identity())
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path
        self.num_experts = 3
        max_threshold = 1/self.num_experts
        self.scaling = 4
        self.lora_gating_ffn = nn.Linear(dim, self.num_experts, bias=False)
        self.lora_threshold_ffn = nn.Linear(dim, 1)
        self.lora_A_ffn = nn.ModuleList(
            [nn.Linear(dim, 8, bias=False) for _ in range(self.num_experts)])
        self.lora_B_ffn = nn.ModuleList(
            [nn.Linear(8, dim, bias=False) for _ in range(self.num_experts)])
        self.lora_dropout_ffn = nn.ModuleList(
            [nn.Dropout(0.1) for _ in range(self.num_experts)])
        experts = nn.ModuleList([LoraExpert(
            self.lora_A_ffn[i],
            self.lora_B_ffn[i],
            self.lora_dropout_ffn[i],
            self.scaling,
        ) for i in range(self.num_experts)])
        self.moe_layer_ffn = AdaMoeLayer(
            experts=experts, gate=self.lora_gating_ffn,
            threshold_fn=self.lora_threshold_ffn, max_threshold=max_threshold)
        
        for i in range (self.num_experts):
            nn.init.kaiming_uniform_(self.lora_A_ffn[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_ffn[i].weight)

    def forward(self, x: Tensor, proto_vector, return_attention=False) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))


        def ffn_residual_func(x: Tensor) -> Tensor:         
            new_ffn=self.moe_layer_ffn(x,proto_vector)          
            return self.ls2(self.mlp(self.norm2(x))+new_ffn)
          
        if return_attention:
            return self.attn(self.norm1(x), return_attn=True)
        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:           
            x = x + self.drop_path1(attn_residual_func(x))       
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:    
            x = x + attn_residual_func(x)           
            x = x + ffn_residual_func(x)
        return x
    
    
def drop_add_residual_stochastic_depth(x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)

def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor

def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor)
    return x_plus_residual

attn_bias_cache: Dict[Tuple, Any] = {}

def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = ([b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list])
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias
    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)
    return attn_bias_cache[all_shapes], cat_tensors

def drop_add_residual_stochastic_depth_list(x_list: List[Tensor], residual_func: Callable[[Tensor, Any], Tensor], sample_drop_ratio: float = 0.0, scaling_vector=None) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]
    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)
    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore
    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs

class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor], self_prompt=None) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)
        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias,self_prompt=self_prompt)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))
            x_list = drop_add_residual_stochastic_depth_list(x_list, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio,
                                                             scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None))
            x_list = drop_add_residual_stochastic_depth_list(x_list, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio,
                                                             scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None))
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias,self_prompt=self_prompt))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list,proto_vector, return_attention=False):
        if isinstance(x_or_x_list, Tensor):
            # Change the following line
            # return super().forward(x_or_x_list)
            return super().forward(x_or_x_list,proto_vector, return_attention)
        elif isinstance(x_or_x_list, list):
            assert (XFORMERS_AVAILABLE), "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError