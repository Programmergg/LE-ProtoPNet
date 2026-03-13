import torch
import torch.nn.functional as F

def sample_gumbel(shape: torch.Size, eps: float = 1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits: torch.Tensor, temperature: float, eps: float = 1e-20):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    g_0 = sample_gumbel(logits.shape).to(logits.device)
    g_1 = sample_gumbel(logits.shape).to(logits.device)
    w_0 = torch.log(torch.clip((1 - logits), 0) + eps)
    w_1 = torch.log(logits + eps)
    numerator = torch.exp((torch.log(logits + eps) + g_1) / temperature)
    denominator = torch.exp((w_0 + g_0) / temperature) + torch.exp((w_1 + g_1) / temperature)
    mask = numerator / (denominator + eps)
    return mask