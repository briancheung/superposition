import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules import BatchNorm1d, BatchNorm2d

from models.linear import complx 


class BaselineNet(nn.Module):
    def __init__(self, input_dim, n_units, activation, n_layers, layer_type):
        super(BaselineNet, self).__init__()
        self.n_units = n_units
        self.activation = activation
        self.n_layers = n_layers
        
        layer_units = [np.prod(input_dim)]
        layer_units += [n_units for i in range(n_layers)]
        layer_units += [10]

        layers = []
        for i in range(len(layer_units)-1):
            layers.append(layer_type(layer_units[i], layer_units[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, time):
        preactivations = []
        r = x
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                r = self.activation(r)
            r = layer(r)
            preactivations.append(r)

        return r, None, preactivations


class SimpleNet(BaselineNet):
    def forward(self, x, time):
        preactivations = []
        r_a, r_b = x, torch.zeros_like(x)
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                r_a = self.activation(r_a)
                r_b = self.activation(r_b)
            r_a, r_b = layer(r_a, r_b)
            preactivations.append(r_a)

        return r_a, r_b, preactivations


class ComplexConvNet(nn.Module):
    def __init__(self, input_dim, n_units, activation, n_layers):
        super(ComplexConvNet, self).__init__()
        self.n_units = n_units
        self.activation = activation
        self.n_layers = n_layers

        self.l = cl.ComplexConv(input_dim[0], n_units, (5,5))
        self.l_middles = nn.ModuleList([cl.ComplexConv(n_units, n_units, (3,3)) for i in range(n_layers)])
        self.l_final = cl.ComplexLinear(n_units, 10)

    def forward(self, x, time):
        preactivations = []
        r_a, r_b = self.l(x, torch.zeros_like(x), time) 
        preactivations.append(r_a)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        for l_middle  in self.l_middles:
            r_a, r_b = l_middle(r_a, r_b, time)
            preactivations.append(r_a)
            r_a = self.activation(r_a)
            r_b = self.activation(r_b)
        r_a = r_a.view(r_a.shape[0], r_a.shape[1], r_a.shape[2]*r_a.shape[3])
        r_b = r_b.view(r_b.shape[0], r_b.shape[1], r_b.shape[2]*r_b.shape[3])
        r_a, r_b = self.l_final(r_a.mean(2), r_b.mean(2))
        return r_a, r_b, preactivations


class ResidualComplexConvNet(ComplexConvNet):
    def __init__(self, input_dim, n_units, activation, n_layers):
        super(ComplexConvNet, self).__init__()
        self.n_units = n_units
        self.activation = activation
        self.n_layers = n_layers

        self.l = cl.ComplexConv(input_dim[0], n_units, (5,5))
        self.l_middles = nn.ModuleList([cl.ComplexConv(n_units, n_units, (3,3), padding=1, init_mult=1e-8) for i in range(n_layers)])
        self.l_final = cl.ComplexLinear(n_units, 10)

    def forward(self, x, time):
        preactivations = []
        r_a, r_b = self.l(x, torch.zeros_like(x), time) 
        preactivations.append(r_a)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        for l_middle  in self.l_middles:
            t_a, t_b = l_middle(r_a, r_b, time)
            r_a = t_a + r_a
            r_b = t_b + r_b
            preactivations.append(r_a)
            r_a = self.activation(r_a)
            r_b = self.activation(r_b)
        r_a = r_a.view(r_a.shape[0], r_a.shape[1], r_a.shape[2]*r_a.shape[3])
        r_b = r_b.view(r_b.shape[0], r_b.shape[1], r_b.shape[2]*r_b.shape[3])
        r_a, r_b = self.l_final(r_a.mean(2), r_b.mean(2))
        return r_a, r_b, preactivations


class OneNet(nn.Module):
    def __init__(self, n_units, activation, n_layers=2):
        super(OneNet, self).__init__()
        self.n_units = n_units
        self.activation = activation
        self.n_layers = n_layers

        self.l = cl.HashLinear(1, n_units, 1, 'hash')
        self.l_sp = cl.HashLinear(n_units, n_units, n_layers, 'hash')
        self.l_final = cl.HashLinear(n_units, 1, 1, 'hash')

    def forward(self, x, time):
        preactivations = []
        r_a, r_b = self.l(x, torch.zeros_like(x), 0)
        preactivations.append(torch.stack((r_a, r_b)))
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        for i in range(self.n_layers):
            r_a, r_b = self.l_sp(r_a, r_b, i)
            preactivations.append(torch.stack((r_a, r_b)))
            r_a = self.activation(r_a)
            r_b = self.activation(r_b)
        r_a, r_b = self.l_final(r_a, r_b, 0)
        return r_a, r_b, preactivations 
