# Adapted From: https://github.com/hynnsk/HP
import torch
import torch.nn as nn
from typing import Optional
from models.backbones.dinov2.utils import create_model

import torch
import torch.nn as nn
from functools import partial



class DinoFeaturizer(nn.Module):
    """
    DinoFeaturizer is a module that extracts features from images using the DINO architecture and a projection head.
    Args:
        arch (str): The architecture name.
        dim (int): The dimension of the output features.
        proj_type (str, optional): The type of projection. Defaults to "nonlinear".
        dropout (float, optional): The dropout rate. Defaults to None.
        pretrained_weights (str, optional): The path to the pretrained weights. Defaults to None.
    """
    def __init__(self, arch: str, dim: int, proj_type: str = "nonlinear", dropout: float | None = None, pretrained_weights: Optional[str] = None):
        super().__init__()
        self.dim = dim
        self.proj_type = proj_type
        self.model = create_model(arch, pretrained_weights)
        #self.patch_size = self.model.patch_size
        self.patch_size = 16 
        self.n_feats = self.model.embed_dim
        if dropout is not None:
            self.dropout = torch.nn.Dropout2d(p=dropout)
        else:
            self.dropout = torch.nn.Dropout2d(p=0)
        self.cluster1 = self.make_clusterer(self.n_feats)
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
       # self.norm = norm_layer(768)

    def make_clusterer(self, in_channels: int):
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer(self, in_channels: int):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
        )
    
    def forward(self, img: torch.Tensor, proto_vector:torch.Tensor):
    
        assert img.shape[2] % self.patch_size == 0
        assert img.shape[3] % self.patch_size == 0
        # get selected layer activations
        feat = self.model.forward_features(img, proto_vector)["x_norm_patchtokens"]
        intermediate_outputs = self.model.get_intermediate_layers(img, proto_vector, 12)   # 25, 196, 768
        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size
        image_feat = feat.reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat
        return (self.dropout(image_feat), self.dropout(code)) ,intermediate_outputs
    
    def get_intermediate_self_attention(self, img: torch.Tensor,proto_vector, return_attentions: bool = True, n: int = 1):
        
        if return_attentions:
            attentions = self.model.get_intermediate_self_attention(img,proto_vector,return_attentions=return_attentions, n=n)
            return attentions

