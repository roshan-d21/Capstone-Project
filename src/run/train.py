import torch
from torch import nn
from torch import optim

from model.Encoder import Encoder
from model.Decoder import Decoder
from model.PoolingModule import PoolingModule

from model.Generator import Generator
from model.Discriminator import Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def main():
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    lr = 0.01
    num_epochs = 100

    # TODO: Load data
    dataloader = None

    generator = Generator().to(device)
    generator.apply(weights_init)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # For each batch
        for i, data in enumerate(dataloader, 0):
            # TODO: (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # TODO: (2) Update G network: maximize log(D(G(z)))
            pass