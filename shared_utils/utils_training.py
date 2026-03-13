import torch
import random
import numpy as np
import torch.nn as nn
from shared_utils.neurokit2_distort_signal_pytorch_tensor import signal_distort

def init_weights(net, init_type="normal", init_gain=0.02):
    """
    Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
    This code has been taken from : https://github.com/OzerCanDevecioglu/Blind-ECG-Restoration-by-Operational-Cycle-GANs
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif (classname.find("BatchNorm2d") != -1):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  # apply the initialization function <init_func>

def batch_reader(batch_petastorm, Lead_to_rec=None):
    signals = batch_petastorm["signal"]
    dev = signals.device
    base_signal = signals.detach().clone()
    if Lead_to_rec is None:
        random_lead = torch.randint(0, 12, (signals.size(0),), device=dev)
    else:
        random_lead = torch.tensor([Lead_to_rec] * signals.size(0), dtype=torch.int64, device=dev)
    input_lead = torch.tensor([[l for l in range(12) if l != random_lead[e].item()] for e in range(random_lead.shape[0])], dtype=torch.int64, device=dev)
    reference_signal = torch.zeros([signals.size(0), 1, signals.size(2)], device=dev)
    for i in range(random_lead.size(0)):
        reference_signal[i, 0, :] = signals[i, random_lead[i], :]
        signals[i, random_lead[i], :] = torch.zeros([1, signals.size(2)])
    return signals, reference_signal, base_signal, random_lead, input_lead

def batch_reader_lead_transform(batch_petastorm, Lead_to_rec=None):
    signals = batch_petastorm["signal"]
    dev = signals.device
    base_signal = signals.clone()
    if Lead_to_rec is None:
        random_lead = torch.randint(0, 12, (signals.size(0),), device=dev)
    else:
        random_lead = torch.tensor([Lead_to_rec] * signals.size(0), dtype=torch.int64, device=dev)
    reference_signal = torch.zeros([signals.size(0), 1, signals.size(2)], device=dev)
    signals = signal_distort(
        signals,
        500,
        noise_amplitude=0.5 * torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        powerline_amplitude=0.5 * torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        artifacts_amplitude=torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        artifacts_number=torch.randint(1, 10, [signals.shape[1], 1], device=dev),
        linear_drift=torch.randint(0, 2, [signals.shape[0], signals.shape[1], 1], device=dev),
        silent=True,
    )
    for i in range(random_lead.size(0)):
        reference_signal[i, 0, :] = signals[i, random_lead[i], :]
        signals[i, random_lead[i], :] = torch.zeros([1, signals.size(2)])
    return signals, reference_signal, random_lead, base_signal

def batch_reader_multilead_nolab(batch_petastorm, n_rec=6, lead_rec=None,):
    signals = batch_petastorm["signal"].type(torch.float32)
    dtype_signal = torch.float32
    base_signal = signals.detach().clone()
    if lead_rec is None:
        lead_rec = torch.randint(1, n_rec + 1, [1]).item()
    random_lead = torch.zeros((signals.size(0), lead_rec), dtype=torch.int64)
    for i in range(random_lead.size(0)):
        random_lead[i, :] = torch.tensor(random.sample(range(12), lead_rec), dtype=torch.int64)  ##This allows sampling without duplicates
    input_lead = torch.tensor([[l for l in range(12) if l not in random_lead[e].tolist()] for e in range(random_lead.size(0))], dtype=torch.int64,)
    reference_signal = torch.zeros([signals.size(0), lead_rec, signals.size(2)], dtype=dtype_signal)
    for i in range(random_lead.size(0)):
        reference_signal[i, :, :] = signals[i, random_lead[i, :], :]
        signals[i, random_lead[i, :], :] = torch.zeros([1, signals.size(2)], dtype=dtype_signal)
    return signals, reference_signal, base_signal, random_lead, input_lead

def batch_reader_multilead(batch_petastorm, lead_rec=None, n_rec=8):
    signals = batch_petastorm["signal"]
    dev = signals.device
    base_signal = signals.clone().detach()
    if lead_rec is None:
        lead_rec = torch.randint(1, n_rec + 1, [1]).item()
    random_lead = torch.zeros((signals.size(0), lead_rec), dtype=torch.int64, device=dev)
    for i in range(random_lead.size(0)):
        random_lead[i, :] = torch.tensor(random.sample(range(12), lead_rec), dtype=torch.int64, device=dev)  ##This allows sampling without duplicates
    input_lead = torch.tensor([[l for l in range(12) if l not in random_lead[e].tolist()] for e in range(random_lead.size(0))], dtype=torch.int64, device=dev)
    reference_signal = torch.zeros([signals.size(0), lead_rec, signals.size(2)], device=dev)
    input_signal = torch.zeros([signals.size(0), signals.size(1) - lead_rec, signals.size(2)], device=dev)
    for i in range(random_lead.size(0)):
        reference_signal[i, :, :] = signals[i, random_lead[i, :], :]
        input_signal[i, :, :] = signals[i, input_lead[i, :], :]
    return input_signal, reference_signal, base_signal, random_lead, input_lead

def pearson_corr(x, y):
    dim = list(range(len(x.shape)))[-1]
    cos = nn.CosineSimilarity(dim=dim, eps=1e-6)
    cost = cos(x - x.mean(dim=dim, keepdim=True), y - y.mean(dim=dim, keepdim=True))
    return cost

def batch_reader_transform_nolab(batch_petastorm, lead_rec=None, n_rec=8):
    # torch.manual_seed(3054)
    signals = batch_petastorm["signal"].cpu()
    dev = signals.device
    base_signal = signals.detach().clone()
    if lead_rec is None:
        lead_rec = torch.randint(1, n_rec + 1, [1]).item()
    random_lead = torch.zeros((signals.size(0), lead_rec), dtype=torch.int64, device=dev)
    for i in range(random_lead.size(0)):
        random_lead[i, :] = torch.tensor(random.sample(range(12), lead_rec), dtype=torch.int64, device=dev)
    input_lead = torch.tensor([[l for l in range(12) if l not in random_lead[e, :]] for e in range(random_lead.size(0))], dtype=torch.int64, device=dev)
    reference_signal = torch.zeros([signals.size(0), lead_rec, signals.size(2)], device=dev)
    signals = signal_distort(
        signals,
        500,
        noise_amplitude=0.5 * torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        powerline_amplitude=0.5 * torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        artifacts_amplitude=torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        artifacts_number=torch.randint(1, 10, [signals.shape[1], 1], device=dev),
        linear_drift=torch.randint(0, 2, [signals.shape[0], signals.shape[1], 1], device=dev),
        silent=True,
    )
    for i in range(random_lead.size(0)):
        reference_signal[i, :, :] = base_signal[i, random_lead[i, :], :]
        signals[i, random_lead[i, :], :] = torch.zeros([1, signals.size(2)], device=dev)
    return signals, reference_signal, base_signal, random_lead, input_lead

def batch_reader_transform(batch_petastorm):
    # torch.manual_seed(3054)
    signals = batch_petastorm["signal"].cpu()
    dev = signals.device
    base_signal = signals.clone()
    lead_rec = torch.randint(1, 5, [1]).item()
    random_lead = torch.zeros((signals.size(0), lead_rec), dtype=torch.int64, device=dev)
    for i in range(random_lead.size(0)):
        random_lead[i, :] = torch.tensor(random.sample(range(12), lead_rec), dtype=torch.int64, device=dev)
    input_lead = torch.tensor([[l for l in range(12) if l not in random_lead[e, :]] for e in range(random_lead.size(0))], dtype=torch.int64, device=dev,)
    reference_signal = torch.zeros([signals.size(0), lead_rec, signals.size(2)], device=dev)
    signals = signal_distort(
        signals,
        500,
        noise_amplitude=0.5 * torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        powerline_amplitude=0.5 * torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        artifacts_amplitude=torch.rand([signals.shape[0], signals.shape[1], 1], device=dev),
        artifacts_number=torch.randint(1, 10, [signals.shape[1], 1], device=dev),
        linear_drift=torch.randint(0, 2, [signals.shape[0], signals.shape[1], 1], device=dev),
        silent=True,
    )
    input_signal = torch.zeros([signals.size(0), signals.size(1) - lead_rec, signals.size(2)], device=dev)
    for i in range(random_lead.size(0)):
        reference_signal[i, :, :] = base_signal[i, random_lead[i, :], :]
        input_signal[i, :, :] = signals[i, input_lead[i, :], :]
    return input_signal, reference_signal, base_signal, random_lead, input_lead

def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    """
    Taken and adapted from : https://discuss.pytorch.org/t/spearmans-correlation/91931/2
    """
    dev = x.device
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp, device=dev)
    if len(x.shape) == 1:
        ranks[tmp] = torch.arange(len(x), device=dev)
    elif len(x.shape) == 2:
        results = torch.arange(x.shape[-1], device=dev) * torch.ones_like(tmp, device=dev)
        for i in range(tmp.shape[0]):
            ranks[i, tmp[i, :]] = results[i, :]
    else:
        results = torch.arange(x.shape[-1], device=dev) * torch.ones_like(tmp, device=dev)
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                ranks[i, j, tmp[i, j, :]] = results[i, j, :]
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """
    Taken and adapted from : https://discuss.pytorch.org/t/spearmans-correlation/91931/2
    Compute correlation between 2 1-D vectors or 2 [batch_size,n_channels,signal_length] vector
    Args:
        x: 1D, 2D or 3D tensors (with the following shape : [Batch_size,n_channels,signal_length])
        y: 1D, 2D or 3D tensors (with the following shape : [Batch_size,n_channels,signal_length])
    Output :
        Spearman correlation {Tensor} : mean Spearman correlation for each batch ([Batch_size,n_channels])
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    n = x.size(-1)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2), dim=-1)
    down = n * (n**2 - 1.0)
    return 1.0 - (upper / down)

def batch_translator(batch_petastorm, Lead_to_rec=None):
    signals = batch_petastorm["signal"]
    if Lead_to_rec is None:
        random_lead = torch.randint(0, 12, (signals.size(0),))
    else:
        random_lead = torch.tensor([Lead_to_rec] * signals.size(0), dtype=torch.int64)
    input_lead = np.array([[l for l in range(signals.shape[1]) if l not in random_lead[e]] for e in range(random_lead.shape[0])])
    X = torch.zeros([signals.shape[0], input_lead.shape[1], signals.shape[2]])
    y = torch.zeros([signals.size(0), 1, signals.size(2)])
    for i in range(random_lead.size(0)):
        y[i, 0, :] = signals[i, random_lead[i], :]
        X[i, :, :] = signals[i, input_lead[i, :], :]
    return X.detach().numpy(), y.detach().numpy(), random_lead.numpy(), input_lead