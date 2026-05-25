import torch
import math

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 1
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                t = state['step']

                grad = p.grad.data

                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                state['exp_avg'] = betas[0] * state['exp_avg'] + (1 - betas[0]) * grad
                state['exp_avg_sq'] = betas[1] * state['exp_avg_sq'] + (1 - betas[1]) * grad * grad
                state['step'] = t + 1

                p.data -= lr * weight_decay * p.data
                p.data -= lr_t * state['exp_avg'] / (torch.sqrt(state['exp_avg_sq']) + eps)

