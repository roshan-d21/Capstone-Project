import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available else 'cpu'

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
    def forward:
        pass
