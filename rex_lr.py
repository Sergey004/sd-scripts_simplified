# Файл: rex_lr.py
# -*- coding: utf-8 -*-
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class REX_LR(_LRScheduler):
    """Reflected Exponential Learning Rate Scheduler."""
    def __init__(self, optimizer, num_epochs, min_val=0.0, max_val=1.0, last_epoch=-1):
        self.num_epochs = num_epochs
        self.min_val = min_val
        self.max_val = max_val
        super(REX_LR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        norm_epoch = epoch / self.num_epochs
        if norm_epoch <= 0.5:
            factor = math.exp(-norm_epoch * 10)
        else:
            factor = math.exp((norm_epoch - 1) * 10)
        factor = factor * (1 - self.min_val / self.max_val) + self.min_val / self.max_val
        return [base_lr * factor for base_lr in self.base_lrs]