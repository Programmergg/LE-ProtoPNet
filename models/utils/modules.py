import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from typing import Optional
import torch.nn.functional as F

class NonNegLinear(nn.Module):
    """
    Applies a linear transformation to the incoming data with non-negative weights.
    In addition, the threshold argument allows setting low importance weights to zero.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to True, adds a learnable bias to the output. Default is True.
        device (str, optional): Device on which to allocate the tensors. Default is None.
        dtype (torch.dtype, optional): Data type of the tensors. Default is None.
        proto_retain (list, optional): List of prototype indices to retain. Default is None.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the parameters of the module.
        """
        custom_sparse_(self.weight, 0.3, std=0.01, mean=0.1)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = torch.abs(self.weight.data)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, threshold: Optional[float] = 0.0, train=True) -> Tensor:
        """
        Performs the forward pass of the module.
        Args:
            input (torch.Tensor): Input tensor.
            threshold (float, optional): Threshold value for setting low importance weights to zero. Default is 0.0.
            train (bool, optional): If set to True, returns the linear transformation output.
                                    If set to False, returns the importance scores and logits. Default is True.
        Returns:
            torch.Tensor: Linear transformation output if train=True, otherwise importance scores and logits.
        """
        self.weight.data.clamp_(0, 5)
        if train:
            return torch.tensor([0]), F.linear(input, self.weight, self.bias)
        else:
            importance = torch.einsum("bp,cp->bpc", input, self.weight)
            importance[importance < threshold] = 0
            if self.bias is not None:
                logits = importance.sum(dim=1) + self.bias
            else:
                logits = importance.sum(dim=1)

            return importance, logits

# from:https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L29
class LayerNorm(nn.Module):
    r"""
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# adapted form the torch library but allow set non-zero mean
def custom_sparse_(tensor, sparsity, std=0.01, mean=0, generator: Optional[torch.Generator] = None):
    r"""
    Fill the 2D input `Tensor` as a sparse matrix.
    The non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via Hessian-free optimization` - Martens, J. (2010).
    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values
        generator: the torch Generator to sample from (default: None)
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))
    with torch.no_grad():
        tensor.normal_(mean, std, generator=generator)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor