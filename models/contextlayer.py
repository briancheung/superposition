import numpy as np
import torch
import torch.nn as nn

from models.linear import complx 


class LayerSPNet(nn.Module):
    def __init__(self, input_dim, n_units, activation, n_layers, learn_key):
        super(LayerSPNet, self).__init__()
        self.n_units = n_units
        self.activation = activation
        self.n_layers = n_layers

        self.l = nn.Linear(np.prod(input_dim), 2*n_units)
        self.l_sp = cl.HashLinear(n_units, n_units, n_layers, 'hash', learn_key=learn_key)
        self.l_final = cl.HashLinear(n_units, 10, 1, 'hash', learn_key=learn_key)

    def forward(self, x, time):
        preactivations = []
        #r_a = self.l(x)
        #r_b = torch.zeros_like(r_a)
        r = self.l(x)
        r_a, r_b = torch.split(r, self.n_units, 1)
        preactivations.append(r_a)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        for i in range(self.n_layers):
            r_a, r_b = self.l_sp(r_a, r_b, i)
            preactivations.append(r_a)
            r_a = self.activation(r_a)
            r_b = self.activation(r_b)
        r_a, r_b = self.l_final(r_a, r_b, 0)
        return r_a, r_b, preactivations 


class LayerSPResNet(LayerSPNet):
    """
    This residual network implementation follows:

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf

    where layers are directly connected to all previous layers.
    """
    def forward(self, x, time):
       preactivations = []
       r = self.l(x)
       r_a, r_b = torch.split(r, self.n_units, 1)
       preactivations.append(r_a)
       r_a = self.activation(r_a)
       r_b = self.activation(r_b)
       for i in range(self.n_layers):
           t_a, t_b = self.l_sp(r_a, r_b, i)
           preactivations = self.activation(r_a)
           r_a = self.activation(t_a) + r_a
           r_b = self.activation(t_b) + t_b
       r_a, r_b = self.l_final(r_a, r_b, 0)
       return r_a, r_b, preactivations


class LayerSPResNet_(LayerSPNet):
    """
    This residual network implementation follows
    the original Residual Network where skip connections
    must pass through a non-linearity.
    """
    def forward(self, x, time):
       preactivations = []
       r = self.l(x)
       r_a, r_b = torch.split(r, self.n_units, 1)
       preactivations.append(r_a)
       r_a = self.activation(r_a)
       r_b = self.activation(r_b)
       for i in range(self.n_layers):
           t_a, t_b = self.l_sp(r_a, r_b, i)
           r_a = t_a + r_a
           r_b = t_b + r_b
           preactivations = self.activation(r_a)
           r_a = self.activation(r_a)
           r_b = self.activation(r_b)
       r_a, r_b = self.l_final(r_a, r_b, 0)
       return r_a, r_b, preactivations


class RecurrentLayerNet(nn.Module):
    def __init__(self, input_dim, n_units, activation, n_layers):
        super(RecurrentLayerNet, self).__init__()
        self.n_units = n_units
        self.activation = activation
        self.n_layers = n_layers

        self.l = nn.Linear(np.prod(input_dim), 2*n_units)
        self.l_sp = cl.HashLinear(n_units, n_units, 1, 'hash')
        self.l_final = cl.HashLinear(n_units, 10, 1, 'hash')

    def forward(self, x, time):
        preactivations = []
        #r_a = self.l(x)
        #r_b = torch.zeros_like(r_a)
        r = self.l(x)
        r_a, r_b = torch.split(r, self.n_units, 1)
        preactivations.append(r_a)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        for i in range(self.n_layers):
            r_a, r_b = self.l_sp(r_a, r_b, 0)
            preactivations.append(r_a)
            r_a = self.activation(r_a)
            r_b = self.activation(r_b)
        r_a, r_b = self.l_final(r_a, r_b, 0)
        return r_a, r_b, preactivations 


class RecurrentLayerResNet(LayerSPNet):
    """
    This residual network implementation follows:

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf

    where layers are directly connected to all previous layers.
    """
    def forward(self, x, time):
       preactivations = []
       r = self.l(x)
       r_a, r_b = torch.split(r, self.n_units, 1)
       preactivations.append(r_a)
       r_a = self.activation(r_a)
       r_b = self.activation(r_b)
       for i in range(self.n_layers):
           t_a, t_b = self.l_sp(r_a, r_b, 0)
           preactivations = self.activation(r_a)
           r_a = self.activation(t_a) + r_a
           r_b = self.activation(t_b) + t_b
       r_a, r_b = self.l_final(r_a, r_b, 0)
       return r_a, r_b, preactivations

