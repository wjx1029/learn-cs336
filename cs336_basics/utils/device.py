import torch

def try_gpu(cuda_index=0):
    if  torch.cuda.is_available() and cuda_index < torch.cuda.device_count():
        return torch.device(f'cuda:{cuda_index}')
    else:
        return torch.device('cpu')