import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
import pytorch_lightning as pl
import torch.nn.functional as F
from matplotlib import pyplot as plt
from typing import Optional as _Optional
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

from models.utils.prune_model import FisherPruningHook
from models.utils.modules import LayerNorm, NonNegLinear
from models.utils.custom_scheduler import CustomLRScheduler
from models.utils.loss import HsLoss, L1Loss, WeightedHsLoss
from models.utils.visualisation import plot_similarity, plot_weight_distribution, plot_weight_heatmap
from torch.autograd import Variable
from typing import Optional


from mfa_analysis  import plot_cluster_tsne

torch.set_float32_matmul_precision("medium")

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        #self.gelu = nn.GELU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x
class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred
    
def l1_regularization(W):
    """
    Calculate the L1 regularization term.
    """
    W_flatten = W.view(W.size(0), -1)
    l1 = torch.norm(W_flatten, 1)
    
    return l1
import torch
from torch import nn, Tensor
import numpy as np
import copy
import random
from tqdm import tqdm
from loralib.loralib.adalora import compute_orth_regu,RankAllocator

class ConfidenceCalculator:
    def compute_confidence_interval(self, data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        stderr = np.std(data, ddof=1) / np.sqrt(n)
        h = stderr * 1.96  # 1.96 corresponds to 95% confidence for a normal distribution
        return mean, h

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        # query: (batch_size, query_len, embed_dim)
        # key, value: (batch_size, key_len, embed_dim)
        attn_output, attn_weights = self.attention(query, key, value)
        return attn_output ,attn_weights
    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt  #zeroslike

        intermediate = []
        attention_scores=[]
        avg_attention_scores=[]

        for layer in self.layers:
            output, attention_score, avg_attention_score = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                attention_scores.append(attention_score)
                avg_attention_scores.append(avg_attention_score)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(attention_scores), torch.stack(avg_attention_scores)

        return output.unsqueeze(0), attention_score, avg_attention_score
    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model=d_model
        self.nhead=nhead


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, attn_output_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, average_attn_weights=False)

        _, attn_avg_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, average_attn_weights=True)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_output_weights, attn_avg_weights

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2, attn_output_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, average_attn_weights=False)#[0]

        _, attn_avg_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, average_attn_weights=True)

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, attn_output_weights, attn_avg_weights

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64 if not s else s
            self.m = 0.6 if not m else m
        if loss_type == 'sphereface':
            self.s = 6.4 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))
            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)

def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    # print(affinity.shape, flat_affinity.shape, probs.shape, avg_probs.flatten().shape)
    # torch.Size([1152, 16384]) torch.Size([1152, 16384]) torch.Size([1152, 16384]) torch.Size([16384])
    loss = sample_entropy - avg_entropy
    return loss

class ClassificationModulePrototype_FSL(pl.LightningModule):
    """
    Classification module  with prototype implementation.
    Args:
        nb_prototypes (int): Number of prototypes.
        ortho_loss_fn (function): Orthogonal loss function.
        prototype_dim (int): Dimension of the prototypes.
        image_encoder (nn.Module): Image encoder model.
        num_classes (int): Number of classes.
        lr (float): Learning rate.
        warmup_epochs (int): Number of warmup epochs.
        loss_type (str, optional): Type of loss function. Defaults to "cross_entropy".
        freeze_backbone (bool, optional): Whether to freeze the backbone. Defaults to False.
        decay_factor (float, optional): Decay factor for learning rate if backbone is not freezed. Defaults to 0.5.
        weight_class (Optional[torch.Tensor], optional): Weight for each class. Defaults to None.
        data_mean (List[float], optional): Mean values of the data. Defaults to [0, 0, 0].
        data_std (List[float], optional): Standard deviation values of the data. Defaults to [1, 1, 1].
        max_epochs (int, optional): Maximum number of epochs. Defaults to 400.
        aggregate_similarity (str, optional): Method for aggregating similarity. Defaults to "nonlinear".
        embed_projection (bool, optional): Whether to use embedding projection. Defaults to True.
        bias_classification_head (bool, optional): Whether to use bias in the classification head. Defaults to False.
        threshold_importance (Optional[float], optional): Threshold for importance in classification head. Defaults to 0.0.
        proto_retain (Optional[float], optional): Retain ratio for prototypes. Defaults to None.
    """
    def __init__(self, nb_prototypes, prototype_dim, image_encoder, num_classes, lr, warmup_epochs, pruning_start,n_way, k_shot, n_query, loss_type="cross_entropy", prune_model_interval=5,
                 freeze_backbone=False, bacbone_lr_multiple=0.001, epsilon=1e-4, weight_class=None, data_mean=[0, 0, 0], data_std=[1, 1, 1], max_epochs=400, aggregate_similarity="nonlinear",
                 embed_projection=True, bias_classification_head=False, threshold_importance: _Optional[float] = 0.0, sparsity_prototype_loss: nn.Module | None = None,
                 sparsity_similarity_loss: nn.Module | None = None, consistency_loss: nn.Module | None = None,info_nce_loss: nn.Module | None = None, cons_loss: nn.Module | None = None,
                 attr_loss: nn.Module | None = None,attn_loss: nn.Module | None = None, l1_reg: nn.Module | None = None, dropblock: nn.Module | None = None, 
                 compact_loss: nn.Module | None = None, diversity_loss: nn.Module | None = None,infoncev1: nn.Module | None = None, cal = ConfidenceCalculator,
                **kwargs):
        super().__init__()
        self.image_encoder = image_encoder
        self.patch_size = self.image_encoder.patch_size
        self.prototype_dim = prototype_dim
        self.freeze_backbone = freeze_backbone
        self.bacbone_lr_multiple = bacbone_lr_multiple
        self.nb_prototypes = nb_prototypes
        self.epsilon = epsilon
        self.lr = lr
        self.prune_model_interval = prune_model_interval
        self.warmup_epochs = warmup_epochs
        self.pruning_start = pruning_start
        self.max_epochs = max_epochs
        self.aggregate_similarity = aggregate_similarity
        self.data_mean = data_mean
        self.data_std = data_std
        self.loss_type = loss_type
        self.prototype_embeddings = nn.Embedding(nb_prototypes, self.prototype_dim)
        self.embed_projection = embed_projection
        self.temperature = torch.tensor(0.1)
        self.threshold_importance = threshold_importance
        nn.init.orthogonal_(self.prototype_embeddings.weight)
        self.save_hyperparameters(logger=False)
        self.dropout = nn.Dropout2d(0.1)
        self.mask_proto = nn.Parameter(torch.ones(nb_prototypes), requires_grad=False)
        self.sparsity_similarity_loss = sparsity_similarity_loss
        self.sparsity_prototype_loss = sparsity_prototype_loss
        self.consistency_loss = consistency_loss
        self.info_nce_loss = info_nce_loss
        self.cons_loss = cons_loss
        self.attr_loss = attr_loss
        self.attn_loss = attn_loss
        self.l1_reg = l1_reg
        self.compact_loss = compact_loss
        self.diversity_loss = diversity_loss
        self.infoncev1 = infoncev1
        self.dropblock = dropblock
        if self.dropblock is not None and self.consistency_loss is not None:
            raise ValueError("Dropblock and consistency loss cannot be used together yet")
        if self.embed_projection:
            self.project_head = nn.Linear(self.prototype_dim, self.prototype_dim)
        if self.aggregate_similarity == "nonlinear":
            self.conv1 = nn.Conv2d(self.nb_prototypes, self.nb_prototypes, (3, 3), padding=1, groups=self.nb_prototypes)
            self.conv2 = nn.Conv2d(self.nb_prototypes, self.nb_prototypes, (1, 1), groups=self.nb_prototypes)
            self.layernorm = LayerNorm(self.nb_prototypes)
        self.classification_head = NonNegLinear(nb_prototypes, num_classes, bias=bias_classification_head)
        if self.loss_type == "binary_cross_entropy":
            metric_collection = MetricCollection(
                [
                    Precision(task="multilabel", num_labels=5),
                    Recall(task="multilabel", num_labels=5),
                ]
            )
            self.classification_loss_fn = F.binary_cross_entropy_with_logits
        elif self.loss_type == "cross_entropy":
            metric_collection = MetricCollection([Accuracy(task="multiclass", num_classes=5)])
            if weight_class is not None:
                self.weight_class = nn.Parameter(torch.tensor(weight_class).float(), requires_grad=False)
            else:
                self.weight_class = None
            self.classification_loss_fn = F.cross_entropy
        self.train_metrics = metric_collection.clone(prefix="train/")
        self.valid_metrics = metric_collection.clone(prefix="val/")
        self.test_metrics = metric_collection.clone(prefix="test/")
        self.img_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.mlp_5 = nn.Linear(512, 312)
        self.mlp_4 = nn.Sequential(
            nn.Linear( 768*3 , 768+ 768*3// 16),
            nn.ReLU(),
            nn.Linear(768+ 768*3// 16, 768) ,
            nn.ReLU(),
            nn.Linear(768, 512)
            )
        self.mlp_3 = nn.Sequential(
            nn.Linear(512 + 768, 512 + 768 // 16),
            nn.ReLU(),
            nn.Linear(512 + 768// 16, 768)
            )
        self.mlp_2 = nn.Sequential(
            nn.Linear(512 + 768, 512 + 768 // 16),
            nn.ReLU(),
            nn.Linear(512 + 768// 16, 768)
            )
        self.mlp_1 = nn.Sequential(
            nn.Linear(512 + 768, 512 + 768 // 16),
            nn.ReLU(),
            nn.Linear(512 + 768// 16, 768)
            )
        self.conv_3 = BasicConv(2, 1, 3, stride=1, padding=(3-1) // 2, relu=False)
        self.conv_2 = BasicConv(2, 1, 3, stride=1, padding=(3-1) // 2, relu=False)
        self.conv_1 = BasicConv(2, 1, 3, stride=1, padding=(3-1) // 2, relu=False)
        self.gelu = nn.GELU()
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.sparse_lambda = 0.1
        self.proto_project = nn.Linear(self.prototype_dim, 3)
       

    def compose_visual(self, obj_feats, att_feats):
        output = torch.cat([obj_feats, att_feats], -1)
        return output

    def backbone_project(self, x: torch.Tensor):
        proto_vector = self.prototype_embeddings.weight.clone().detach()
        proto_vector = self.proto_project(proto_vector)

        model_output,inter = self.image_encoder(x,proto_vector)
        model_output = model_output[1]  # (B,512,16,16)

        feat_layer10=inter[2].permute(0, 2, 1).reshape(inter[2].shape[0],inter[2].shape[2],14,14)
        feat_layer11=inter[5].permute(0, 2, 1).reshape(inter[5].shape[0],inter[5].shape[2],14,14)
        feat_layer12=inter[8].permute(0, 2, 1).reshape(inter[8].shape[0],inter[8].shape[2],14,14)

        vis_obj_l9 = model_output
        vis_obj_l3 = model_output
        vis_obj_l6 = model_output

        relation1 = torch.cat((model_output, feat_layer12), dim=1)  
        c_scale = F.sigmoid(self.mlp_3(self.img_avg_pool(relation1).squeeze())).unsqueeze(2).unsqueeze(3).expand_as(feat_layer12)
        relation1_inter = feat_layer12 * c_scale
        relation1_s = torch.cat((torch.mean(vis_obj_l9, 1).unsqueeze(1), torch.mean(relation1_inter, 1).unsqueeze(1)), dim=1)
        s_scale1 = F.sigmoid(self.conv_3(relation1_s))
       
        img_feat_l9 = feat_layer12 + relation1_inter * s_scale1

        relation2 = torch.cat((model_output, feat_layer11), dim=1)
        c_scale = F.sigmoid(self.mlp_2(self.img_avg_pool(relation2).squeeze())).unsqueeze(2).unsqueeze(3).expand_as(feat_layer11)
        relation2_inter = feat_layer11 * c_scale
        relation2_s = torch.cat((torch.mean(vis_obj_l6, 1).unsqueeze(1), torch.mean(relation2_inter, 1).unsqueeze(1)), dim=1)
        s_scale2 = F.sigmoid(self.conv_2(relation2_s))
 
        img_feat_l6 = feat_layer11 + relation2_inter * s_scale2
        
        relation3 = torch.cat((model_output, feat_layer10), dim=1) 
        c_scale = F.sigmoid(self.mlp_1(self.img_avg_pool(relation3).squeeze())).unsqueeze(2).unsqueeze(3).expand_as(feat_layer10)
        relation3_inter = feat_layer10 * c_scale
        relation3_s = torch.cat((torch.mean(vis_obj_l3, 1).unsqueeze(1), torch.mean(relation3_inter, 1).unsqueeze(1)), dim=1)
        s_scale3 = F.sigmoid(self.conv_1(relation3_s)) 
        
        img_feat_l3 = feat_layer10 + relation3_inter * s_scale3
    
        fused_vis = torch.cat((img_feat_l6, img_feat_l9),dim=1) #768*3
        fused_vis = torch.cat((fused_vis, img_feat_l3), dim=1) #768*4
        fused_vis = fused_vis.permute(0, 2, 3, 1) # (B,,16,16,1024*3)
        attr_embed = self.mlp_4(fused_vis)  #  torch.Size([128, 14, 14, 512]
        attr_pred = attr_embed.reshape(x.shape[0],-1,512).mean(dim=1) # (B,512)
        attr_pred = self.mlp_5(attr_pred) # (B,312)
        attr_pred = F.normalize(attr_pred, dim=-1)

        if self.dropblock is not None:
            model_output = self.dropblock(model_output)
        modeloutput_s = model_output.permute(0, 2, 3, 1).reshape(-1, self.prototype_dim)
        if self.embed_projection:
            modeloutput_s = self.gelu(modeloutput_s)
            modeloutput_z = self.project_head(modeloutput_s)
        else:
            modeloutput_z = modeloutput_s
        modeloutput_z = modeloutput_z.reshape(x.shape[0], -1, self.prototype_dim)
        modeloutput_z = F.normalize(modeloutput_z, dim=-1)

        modeloutput_z = modeloutput_z 

        return modeloutput_z ,attr_pred ,feat_layer10, feat_layer11, feat_layer12

    def forward(self, x: torch.Tensor, train=False):
        modeloutput_z, attr_pred ,img_feat_l3, img_feat_l6, img_feat_l9 = self.backbone_project(x)
        similarity = torch.einsum(
            "bic, bnc -> bni",
            F.normalize(modeloutput_z.float(), dim=-1),
            F.normalize(self.prototype_embeddings.weight.unsqueeze(0).repeat(x.shape[0], 1, 1), dim=-1),
        )  #(B, 196, 300)
        
        similarity = F.softmax(similarity / (self.temperature), dim=1)
        similarity_prototypes = similarity * self.mask_proto[None, :, None]
        similarity_background = similarity * (1 - self.mask_proto[None, :, None])
        similarity_ratio = similarity_background.sum() / similarity.sum()
        self.log("similarity_ratio", similarity_ratio)
        if self.aggregate_similarity == "nonlinear":
            h = w = int(similarity.shape[-1] ** 0.5)
            similarity_reshaped = rearrange(similarity_prototypes, "b n (h w) -> b n h w", h=h, w=w)
            similarity_1 = self.conv1(similarity_reshaped)
            similarity_2 = self.conv2(similarity_reshaped)
            similarity_reshaped = similarity_1 + similarity_2
            similarity_reshaped = similarity_reshaped.permute(0, 2, 3, 1)
            similarity_reshaped = self.layernorm(similarity_reshaped)
            similarity_reshaped = similarity_reshaped.permute(0, 3, 1, 2)
            similarity_score = self.gelu(similarity_reshaped)
            similarity_score = similarity_reshaped.amax(dim=(2, 3))  # (B,N)
        else:
            similarity_score = similarity_prototypes.amax(dim=-1)
        total_out_dict = {   
            "similiarity_score": similarity_score,
            "projected_proto": self.prototype_embeddings.weight,
            "similarity_prototype": similarity_prototypes,
            "similarity_background": similarity_background,
            "similarity_score": similarity_score,
            "attr_embed": attr_pred,
            "img_feat_l3": img_feat_l3,
            "img_feat_l6": img_feat_l6,
            "img_feat_l9": img_feat_l9,     
        }
        return total_out_dict

    def configure_optimizers(self):
        if self.freeze_backbone:
            for p in self.image_encoder.model.parameters():
                p.requires_grad = False
            self.image_encoder.model.eval()
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        else:
            param_groups = []
            param_other = []

            for name, param in self.image_encoder.named_parameters():
                if 'lora' not in name :
                    param.requires_grad = False
                else:
                    lr = self.lr * (self.bacbone_lr_multiple)
                    param_groups.append({"params": param, "lr": lr})
               

    
            for param in self.named_parameters():
                if "image_encoder.model." not in param[0]:
                    param_groups.append({"params": param[1], "lr": self.lr})
                  
            optimizer = optim.AdamW(param_groups)
             
        scheduler_custom = CustomLRScheduler(optimizer, warmup_epochs=self.warmup_epochs, min_lr=1e-4, max_epochs=300)
        return [optimizer], [scheduler_custom] 
    
    def _calculate_loss(self, batch, mode="train", plot_similarity=False,mask_proto=None):
        train = True if mode == "train" else False
        if len(batch) == 4:
            imgs, labels, img_aug = batch
            imgs_support, labels_support, img_aug_support = imgs[:self.n_way*self.k_shot], labels[:self.n_way*self.k_shot], img_aug[:self.n_way*self.k_shot]
            imgs_query, labels_query, img_aug_query = imgs[self.n_way*self.k_shot:], labels[self.n_way*self.k_shot:], img_aug[self.n_way*self.k_shot:]
           
            out_dict = self.forward(imgs,train)
            out_dict_support = self.forward(imgs_support, train)
            out_dict_query = self.forward(imgs_query, train)
            modeloutput_z_aug = self.backbone_project(img_aug)
            if self.consistency_loss is not None:
                loss_consistency = self.consistency_loss(out_dict["modeloutput_z"], modeloutput_z_aug)  # 原图与增强后的图
            else:
                loss_consistency = torch.tensor(0, device=self.device)
        else:
            imgs, labels  = batch
            imgs_support, labels_support = imgs[:self.n_way*self.k_shot], labels[:self.n_way*self.k_shot]#, attr[:self.n_way*self.k_shot]
            imgs_query, labels_query = imgs[self.n_way*self.k_shot:], labels[self.n_way*self.k_shot:]#, attr[self.n_way*self.k_shot:]
            out_dict_support = self.forward(imgs_support, train)
            out_dict_query = self.forward(imgs_query, train)
            loss_consistency = torch.tensor(0, device=self.device)
       
        
        preds_support =out_dict_support["similiarity_score"] + out_dict_support["attr_embed"]
        preds_query = out_dict_query["similiarity_score"] + out_dict_query["attr_embed"]
      
        combined_preds = torch.einsum(
            "qp, sp -> qs",
            preds_query,
            preds_support,
        ) # (nway * nquery, nway * kshot)

        importance = preds_support.T
        importance = importance.reshape(312, self.n_way, self.k_shot)
        importance = importance.mean(dim=-1)
        importance[importance < 0] = 0
       
        combined_similarity = combined_preds.view(self.n_way * self.n_query, self.n_way, self.k_shot)
        class_similarity = (combined_similarity.mean(dim=-1)) # (nway * nquery, nway)
        labels_query_onehot = Variable(torch.arange(0, self.n_way).view(self.n_way, 1).expand(self.n_way, self.n_query).long().cuda(), requires_grad=False).reshape(-1)
       
        if self.loss_type == "binary_cross_entropy":
            labels = labels.float()
            
        elif self.loss_type == "cross_entropy":  #cross_entropy
            loss_classification = self.classification_loss_fn(class_similarity, labels_query_onehot, weight=self.weight_class)
        self.log(f"{mode}/loss_classification", loss_classification)
        loss = loss_classification

        if self.info_nce_loss is not None:
            info_nce_loss = self.info_nce_loss(out_dict_query["similarity_prototype"], out_dict_query["similarity_background"])
            loss += info_nce_loss
            self.log(f"{mode}/loss_info_nce", info_nce_loss)
          
                    

        similarity = (out_dict_query["similarity_prototype"] + out_dict_query["similarity_background"])
        similarity_tmp = rearrange(similarity, "b p n -> (b n) p")
        l_t = -(torch.log(torch.tanh(torch.sum(similarity_tmp, dim=0)) + 1e-20).mean()) # 
        self.log(f"{mode}/loss_t", l_t)
        loss += l_t
        loss += (torch.clip((torch.tensor(self.current_epoch / self.warmup_epochs)), 0, 1) * loss_consistency)
        self.log(f"{mode}/loss_consistency", loss_consistency)
        if isinstance(self.sparsity_prototype_loss, WeightedHsLoss):
            similarity_score = out_dict_query["similarity_score"]
            sparsity_prototype_loss = self.sparsity_prototype_loss(self.classification_head.weight * self.mask_proto[None, :], similarity_score)
        elif isinstance(self.sparsity_prototype_loss, L1Loss):
            sparsity_prototype_loss = self.sparsity_prototype_loss(self.classification_head.weight)
        else:
            sparsity_prototype_loss = torch.tensor(0, device=self.device)
        # if self.current_epoch > self.warmup_epochs:
        loss += (torch.clip((torch.tensor(self.current_epoch / self.warmup_epochs)), 0, 1) * sparsity_prototype_loss)
        self.log(f"{mode}/sparsity_prototype_loss", sparsity_prototype_loss)
        if isinstance(self.sparsity_similarity_loss, HsLoss):
            similarity_prototype = out_dict["similarity_prototype"]
            sparsity_similarity_loss = self.sparsity_similarity_loss(similarity_prototype)
            if self.current_epoch > self.warmup_epochs:
                loss += sparsity_similarity_loss
                self.log(f"{mode}/sparsity_similarity_loss", sparsity_similarity_loss)          
        else:
            sparsity_similarity_loss = torch.tensor(0, device=self.device)
        self.log("%s/loss" % mode, loss)
        if mode == "train":
            metrics = self.train_metrics.update(class_similarity, labels_query_onehot)
          
        elif mode == "val":
            metrics = self.valid_metrics.update(class_similarity, labels_query_onehot)
           
        elif mode == "test":
            print(class_similarity.shape, labels_query_onehot.shape)
            metrics = self.test_metrics.update(class_similarity, labels_query_onehot)
            
        if plot_similarity:
            self.plot_signal(imgs_query, out_dict_query , importance , combined_preds)  #combined_preds(75,25)
        return loss, metrics 

    def training_step(self, batch, batch_idx):
        loss, metrics = self._calculate_loss(batch, mode="train",mask_proto=self.mask_proto)
        return loss

    def on_train_epoch_start(self):
        self.fisher_hook = FisherPruningHook(self.classification_head)

    def on_train_epoch_end(self):
        output = self.train_metrics.compute()
        self.log_dict(output)
        self.train_metrics.reset()
      

    def on_validation_epoch_end(self):
        output = self.valid_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.valid_metrics.reset()

    def validation_step(self, batch, batch_idx):
        plot_similarity = True if batch_idx == 0 else False
        self._calculate_loss(batch, mode="val", plot_similarity=plot_similarity)

    def test_step(self, batch, batch_idx):
        plot_similarity = True if batch_idx == 0 else False
        self._calculate_loss(batch, mode="test")


     
    
    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output, prog_bar=True)
        self.test_metrics.reset()

    @rank_zero_only
    def plot_signal(self, imgs, pred_dict,importance,pred):
        fig1 = plot_similarity(imgs, pred_dict, importance,pred,torch.tensor(self.data_mean), torch.tensor(self.data_std), fig_nb= 75, nb_proto_plot=10)
        fig2 = plot_weight_distribution(self.classification_head.weight.detach().cpu().numpy())
        fig3 = plot_weight_heatmap((self.classification_head.weight * self.mask_proto[None, :]).detach().cpu().numpy())
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.logger.experiment.add_figure(f"plot_similarity", fig1, self.global_step)
            self.logger.experiment.add_figure(f"plot_weight_distribution", fig2, self.global_step)
            self.logger.experiment.add_figure(f"plot_weight_heatmap", fig3, self.global_step)
        elif isinstance(self.logger, pl.loggers.WandbLogger):
            wandb.log({f"Sample": fig1})
            wandb.log({f"weight_distribution": wandb.Image(fig2)})
            wandb.log({f"weight_heatmap": wandb.Image(fig3)})
        plt.close("all")