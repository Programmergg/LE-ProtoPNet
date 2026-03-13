import math
import warnings
from torch.optim.lr_scheduler import LRScheduler

class CustomLRScheduler(LRScheduler):
    """
    Custom learning rate scheduler that implements a half-cycle cosine decay
    after warmup where the lr is linearly increased.
    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        warmup_epochs (int): The number of warmup epochs.
        min_lr (float): The minimum learning rate.
        max_epochs (int): The maximum number of epochs.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
        verbose (bool, optional): If True, prints a message for each update. Default: False.
    """
    def __init__(self, optimizer, warmup_epochs: int,  min_lr: float, max_epochs: int, last_epoch=-1, verbose=False,):
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)
        if self.last_epoch < self.warmup_epochs:
            return [group.get("lr_scale", 1) * base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        else:
            return [group.get("lr_scale", 1) * (self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))))
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]