import torch
from torch import nn

from model.Encoder import Encoder
from model.Decoder import Decoder
from model.PoolingModule import PoolingModule

device = 'cuda' if torch.cuda.is_available else 'cpu'

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.pool = PoolingModule()
        self.decoder = Decoder()
    
    def forward(self):
        pass
