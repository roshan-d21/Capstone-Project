import torch
from torch import nn

from model.Encoder import Encoder

device = 'cuda' if torch.cuda.is_available else 'cpu'

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = Encoder()
    
    def forward(self):
        pass
