"""
From https://github.com/stevenstalder/NN-Explainer/blob/main/utils/loss.py
"""
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

def norm(t):
    return F.normalize(t, dim=-1, eps=1e-10)

def tensor_correlation(a, b):
    return torch.einsum("nhc,nic->nhi", a, b)

class WeightedHsLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, weight_loss: float = 1.0, structured: bool = True, compute_importance: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight_loss = weight_loss
        self.structured = structured
        self.compute_importance = compute_importance

    def forward(self, classification_head_weight: torch.Tensor, similarity_score: torch.Tensor = None, eps: float = 1e-20) -> torch.Tensor:
        if self.compute_importance:
            importance = torch.einsum("bp,cp->bpc", similarity_score, classification_head_weight)
            if self.structured:
                sparsity_loss = (self.alpha * (torch.sum(torch.norm(importance, p=2, dim=1), dim=1) ** 2) + self.beta *
                                 (torch.sum(torch.norm(importance, p=2, dim=2), dim=1) ** 2)) / (torch.norm(importance, p=2, dim=[1, 2]) ** 2 + 1e-20)
                sparsity_loss = sparsity_loss / torch.numel(importance[0]) ** 0.5
                sparsity_loss += self.gamma * torch.norm(importance, dim=[1, 2], p=2)
                sparsity_loss = sparsity_loss.mean()
            else:
                sparsity_loss = self.alpha * (torch.norm(importance, p=1, dim=[1, 2]) ** 2 / torch.sum(importance**2 + eps, dim=[1, 2]))
                sparsity_loss = sparsity_loss / torch.numel(importance[0]) ** 0.5
                sparsity_loss += self.gamma * torch.norm(importance, dim=[1, 2], p=2)
                sparsity_loss = sparsity_loss.mean()
        else:
            # classification head weight dim are inversed so inverse beta and alpha
            if self.structured:
                sparsity_loss = (self.beta * (torch.sum(torch.norm(classification_head_weight, p=2, dim=0), dim=0) ** 2) + self.alpha *
                                 (torch.sum(torch.norm(classification_head_weight, p=2, dim=1), dim=0) ** 2)) / (torch.norm(classification_head_weight, p=2) ** 2 + 1e-20)
                sparsity_loss = (sparsity_loss / torch.numel(classification_head_weight) ** 0.5)
                sparsity_loss += self.gamma * torch.norm(classification_head_weight, p=2)
            else:
                sparsity_loss = self.alpha * (torch.norm(classification_head_weight, p=1) ** 2 / torch.sum(classification_head_weight**2 + eps))
                sparsity_loss = (sparsity_loss / torch.numel(classification_head_weight) ** 0.5)
                sparsity_loss += self.gamma * torch.norm(classification_head_weight, p=2)
        return self.weight_loss * sparsity_loss

class HsLoss(nn.Module):
    def __init__(self, weight_loss: float = 1.0,):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, input: torch.Tensor, ) -> torch.Tensor:
        sparsity_loss = (torch.norm(input + 1e-20, dim=[1, 2], p=1) ** 2 / torch.sum(input + 1e-20**2, dim=[1, 2])).mean()
        sparsity_loss = sparsity_loss / torch.numel(input[0]) ** 0.5
        return self.weight_loss * sparsity_loss

class L1Loss(nn.Module):
    def __init__(self, weight_loss: float = 1.0, **kwargs):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, classification_head_weight) -> torch.Tensor:
        sparsity_loss = torch.norm(classification_head_weight, p=1)
        sparsity_loss = sparsity_loss / torch.numel(classification_head_weight) ** 0.5
        return self.weight_loss * sparsity_loss

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07, loss_version: int = 1, reweighting: int = 1, weight_loss: float = 1.0,):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.loss_version = loss_version
        self.reweighting = reweighting
        self.weight_loss = weight_loss

    def forward(self, modeloutput_z: torch.Tensor, prototypes: torch.Tensor, rho: float = 0.02):
        device = torch.device("cuda") if modeloutput_z.is_cuda else torch.device("cpu")
        prototypes = F.normalize(prototypes.float(), dim=-1)
        modeloutput_z = F.normalize(modeloutput_z, dim=-1)
        spatial_size = modeloutput_z.shape[1]
        modeloutput_z = modeloutput_z.reshape(-1, modeloutput_z.shape[-1])
        batch_size = modeloutput_z.shape[0]
        split = int(spatial_size)
        mini_iters = int(batch_size / split)
        negative_mask_one = torch.scatter(torch.ones((split, batch_size)), 1, torch.arange(split).view(-1, 1), 0).to(device)
        mask_neglect_base = torch.FloatTensor(split, batch_size).uniform_() < rho
        mask_neglect_base = mask_neglect_base.cuda()
        loss = torch.tensor(0).to(device)
        Rpoint = torch.matmul(modeloutput_z, prototypes.transpose(0, 1))
        Rpoint = torch.max(Rpoint, dim=1).values
        Rpoint_T = Rpoint.unsqueeze(-1).repeat(1, split)
        for mi in range(mini_iters):
            modeloutput_z_one = modeloutput_z[mi * split : (mi + 1) * split]
            output_cossim_one = torch.matmul(modeloutput_z_one, modeloutput_z.transpose(0, 1))
            output_cossim_one_T = output_cossim_one.transpose(0, 1)
            mask_one_T = Rpoint_T < output_cossim_one_T
            mask_one_T = torch.tensor(mask_one_T.transpose(0, 1))
            Rpoint_one = Rpoint[mi * split : (mi + 1) * split]
            Rpoint_one = Rpoint_one.unsqueeze(-1).repeat(1, batch_size)
            mask_one = torch.tensor((Rpoint_one < output_cossim_one))
            mask_one = torch.logical_or(mask_one, mask_one_T)
            neglect_mask = torch.logical_or(mask_one, mask_neglect_base)
            neglect_negative_mask_one = negative_mask_one * neglect_mask
            mask_one = mask_one * negative_mask_one
            modeloutput_z_one = modeloutput_z[mi * split : (mi + 1) * split]
            anchor_dot_contrast_one = torch.div(torch.matmul(modeloutput_z_one, modeloutput_z.T), self.temperature)
            logits_max_one, _ = torch.max(anchor_dot_contrast_one, dim=1, keepdim=True)
            logits_one = anchor_dot_contrast_one - logits_max_one.detach()
            exp_logits_one = torch.exp(logits_one) * neglect_negative_mask_one
            log_prob_one = logits_one - torch.log(exp_logits_one.sum(1, keepdim=True))
            if self.loss_version == 1:
                nonzero_idx = torch.where(mask_one.sum(1) != 0.0)
                mask_one = mask_one[nonzero_idx]
                log_prob_one = log_prob_one[nonzero_idx]
                # mask_ema_one = mask_ema_one[nonzero_idx]
                weighted_mask = mask_one.detach()
                if self.reweighting == 1:
                    pnm = torch.tensor(torch.sum(weighted_mask, dim=1), dtype=torch.float32)
                    pnm = pnm / torch.sum(pnm)
                    pnm = pnm / torch.mean(pnm)
                else:
                    pnm = 1
                mean_log_prob_pos_one = (weighted_mask * log_prob_one).sum(1) / (weighted_mask.sum(1))
                loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob_pos_one * pnm)
            elif self.loss_version == 2:
                nonzero_idx = torch.where(mask_one.sum(1) != 0.0)
                mask_one = mask_one[nonzero_idx]
                if self.reweighting == 1:
                    pnm = torch.tensor(torch.sum(mask_one, dim=1), dtype=torch.float32)
                    pnm = pnm / torch.sum(pnm)
                    pnm = pnm / torch.mean(pnm)
                else:
                    pnm = 1
                mean_log_prob_pos_one = (mask_one * log_prob_one[nonzero_idx]).sum(1) / (mask_one.sum(1))
                loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob_pos_one * pnm)
        return loss / mini_iters * self.weight_loss

class WeightedOrthoLoss(nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, prototypes: torch.Tensor, weight_class: torch.Tensor):
        ortho_loss_proto = torch.matmul(prototypes, prototypes.transpose(0, 1)) - torch.eye(prototypes.shape[0], device=prototypes.device)
        weighted_ortho = weight_class.unsqueeze(-1) * ortho_loss_proto.unsqueeze(0)
        ortho_loss_proto = torch.norm(weighted_ortho, dim=[1, 2], p=2).mean()
        return self.weight_loss * ortho_loss_proto


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, similarity: torch.Tensor):
        # Subtract max similarity from all other similaritie
        max_similarity, idx_max = similarity.max(dim=-1, keepdim=True)
        contrastive_similarity = similarity - max_similarity
        # Apply softmax to get probability distribution over prototypes
        prob = F.softmax(contrastive_similarity, dim=-1)
        test_prob = rearrange(prob, "b c p -> (b c) p")
        # Compute negative log likelihood of correct prototype
        loss = F.nll_loss(test_prob.log(), idx_max.reshape(-1))
        return self.weight_loss * loss


class ConsistencyLoss(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss
        self.fn = nn.PairwiseDistance()

    def forward(self, embeddings: torch.Tensor, embeddings_aug: torch.Tensor):
        # Subtract max similarity from all other similaritie
        loss_consistency = torch.mean(self.fn(embeddings, embeddings_aug))
        return self.weight_loss * loss_consistency


class Info_nce_loss(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, similarity_proto,similarity_background,  temperature=0.15):
        combined_similarity = torch.cat([similarity_proto, similarity_background], dim=1)

        combined_similarity /= temperature
        exp_logits = torch.exp(combined_similarity)
        exp_positives = torch.exp(similarity_proto)
        numerator = torch.sum(exp_positives, dim=1)
        denominator = torch.sum(exp_logits, dim=1)
        epsilon = 1e-8 
        loss = -torch.log((numerator + epsilon) / (denominator + epsilon))
        info_nce_loss = torch.mean(loss)

        return info_nce_loss*self.weight_loss
        
class cons_loss(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()  
        self.weight_loss = weight_loss

    def forward(self, cons_loss):    
        return self.weight_loss * cons_loss

class attr_loss(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()  
        self.weight_loss = weight_loss

    def forward(self, attr_pred, attr):    
        loss_fn = nn.BCEWithLogitsLoss()
        attr = attr.float()
        return self.weight_loss * loss_fn(attr_pred, attr)


# class attn_loss(torch.nn.Module):
#     def __init__(self, weight_loss: float = 1.0):
#         super().__init__()  
#         self.weight_loss = weight_loss
    

#     def unravel_index(self,index, shape):
#         rows = index // shape[1]
#         cols = index % shape[1]
#         return rows, cols

#     def forward(self,  A_m):
#         B, _, W, H = A_m.shape  # 获取 batch size 和空间维度
#         loss = 0.0
#         for b in range(B):
#             A_m_b = A_m[b, 0]  # 提取第 b 个样本的 14x14 矩阵
#             max_idx = torch.argmax(A_m_b)  # 找到最大值的索引（展开后的一维索引）
#             w_star, h_star = self.unravel_index(max_idx, (W, H))  # 获取二维坐标
#             for w in range(W):
#                 for h in range(H):
#                     w_diff = (w - w_star) ** 2
#                     h_diff = (h - h_star) ** 2
#                     loss += A_m_b[w, h] * (w_diff + h_diff)
#         return loss*self.weight_loss/B   

class attn_loss(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self, A_m):
        B, _, W, H = A_m.shape  # 获取 batch size 和空间维度
        # 获取每个样本的最大值索引
        A_m_b = A_m[:, 0, :, :]  # 提取所有样本的 14x14 矩阵，形状为 (B, W, H)
        max_idx = torch.argmax(A_m_b.view(B, -1), dim=1)  # 每个样本展开后的一维索引，形状为 (B,)
        w_star = max_idx // W
        h_star = max_idx % H

        # 生成坐标网格
        w_coords = torch.arange(W, device=A_m.device).view(1, W, 1)
        h_coords = torch.arange(H, device=A_m.device).view(1, 1, H)

        # 计算坐标差的平方
        w_diff = (w_coords - w_star.view(B, 1, 1)) ** 2
        h_diff = (h_coords - h_star.view(B, 1, 1)) ** 2

        # 计算损失
        diff = w_diff + h_diff  # 形状为 (B, W, H)
        loss = (A_m_b * diff).sum() * self.weight_loss / B

        return loss


class l1_reg (torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()  
        self.weight_loss = weight_loss

    def forward(self, para):
        
        return self.weight_loss * para
    

class CompactnessLoss(torch.nn.Module):
    def __init__(self, zeta=0.5):
        super().__init__()
        self.zeta = zeta
    def forward(self, similarity_maps):
        B, N, H, W = similarity_maps.shape
        max_indices = similarity_maps.view(B, N, -1).argmax(dim=2)  # shape: (B, N)
        w_star = max_indices // W
        h_star = max_indices % W
        w_coords = torch.arange(H, device=similarity_maps.device).view(1, 1, H, 1)
        h_coords = torch.arange(W, device=similarity_maps.device).view(1, 1, 1, W)
        dist_w = (w_coords - w_star.view(B, N, 1, 1)) ** 2
        dist_h = (h_coords - h_star.view(B, N, 1, 1)) ** 2

        # 计算紧凑损失并求平均
        compactness_loss = (similarity_maps * (dist_w + dist_h)).sum() / (B * N)

        return self.zeta * compactness_loss 
    
class DiversityLoss(torch.nn.Module):    
    def __init__(self, zeta=0.5):
        super().__init__()
        self.zeta = zeta

    def forward(self, similarity_maps):
        B, N, H, W = similarity_maps.shape
        loss_div = 0.0
        for m in range(N):
            A_m = similarity_maps[:, m, :, :]  # 当前相似度图
            other_maps = torch.cat([similarity_maps[:, :m, :, :], similarity_maps[:, m+1:, :, :]], dim=1)
            max_other_maps = other_maps.max(dim=1).values  # shape: (B, H, W)
            diff = F.relu(max_other_maps - self.zeta) * A_m
            loss_div += torch.sum(diff)

        return self.zeta * loss_div / (B * N * H * W)

class info_nce_loss_v1(torch.nn.Module):
    def __init__(self, weight_loss: float = 1.0):
        super().__init__()
        self.weight_loss = weight_loss

    def forward(self,  query_similarity_matrix, support_similarity_matrix, query_labels, support_labels, temperature=0.07):
        similarity_matrix = torch.mm(query_similarity_matrix, support_similarity_matrix.t())  # (75, 25)
        positive_mask = query_labels.unsqueeze(1) == support_labels.unsqueeze(0)  # (75, 25)
        negative_mask = ~positive_mask  # 反转掩码，作为负样
        similarity_matrix /= temperature

        # 获取正样本对的相似度
        positive_samples = similarity_matrix[positive_mask].view(75, -1)  # 每个query的正样本
        negative_samples = similarity_matrix[negative_mask].view(75, -1)  # 每个query的负样本

        # 计算 InfoNCE loss
        positive_loss = torch.logsumexp(positive_samples, dim=-1)  # 正样本的loss
        negative_loss = torch.logsumexp(torch.cat([positive_samples, negative_samples], dim=-1), dim=-1)
        infonce_loss = (negative_loss - positive_loss).mean()
        
        return self.weight_loss * infonce_loss
    

# class attr_loss(nn.Module):
#     def __init__(self, weight_loss: float = 1.0):
#         super().__init__()
#         self.weight_loss = weight_loss

#     def forward(self, a_pred, a_true):
#         B, A = a_true.shape  # B: 样本数，A: 属性数
#         n_pos = torch.sum(a_true, dim=1, keepdim=True)  # 每个样本属性存在的数量
#         w_pos = 1.0 / torch.clamp(n_pos, min=1.0)  # 避免除零
#         n_neg = torch.sum(1 - a_true, dim=1, keepdim=True)  # 每个样本属性不存在的数量
#         w_neg = 1.0 / torch.clamp(n_neg, min=1.0)  # 避免除零
#         log_prob_pos = a_true * torch.log(a_pred + 1e-8)  # 避免 log(0)
#         log_prob_neg = (1 - a_true) * torch.log(1 - a_pred + 1e-8)
#         weighted_loss_pos = w_pos * log_prob_pos  # 正例部分
#         weighted_loss_neg = w_neg * log_prob_neg  # 负例部分
#         loss = -torch.sum(weighted_loss_pos + weighted_loss_neg) / B  # 对 batch 求平均
#         return loss*self.weight_loss
