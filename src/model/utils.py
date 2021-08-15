import torch
import torch.nn as nn

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    '''Create and return multilayer perceptron'''
    layers = []

    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))

        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())

        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    '''Generate noise to be added to input'''
    
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()

    raise ValueError('Unrecognized noise type "%s"' % noise_type)