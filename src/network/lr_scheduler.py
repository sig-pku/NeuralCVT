import math
import torch

class LR_Scheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self,optimizer,num_epochs: int,decay_type: str,warmup_ratio: float):
        def lr_lambda(epoch: int):
            warmup_epochs = int(warmup_ratio * num_epochs)
            warmup_init_lambda = 0.1
            if epoch < warmup_epochs:
                return warmup_init_lambda + (1.0 - warmup_init_lambda) * (float(epoch) / float(warmup_epochs))

            if decay_type == 'constant':
                return 1.0
            elif decay_type == 'cosine':
                progress = float(epoch - warmup_epochs) / float(num_epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                raise ValueError("Unknown decay_type: {}".format(decay_type))

        super().__init__(optimizer, lr_lambda)