import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from  models.linear import (complx,
                            real)


class HashNet(nn.Module):
    def __init__(self, input_dim, n_units,
                 activation, n_layers,
                 period, key_pick, 
                 learn_key, layer_type):
        super(HashNet, self).__init__()
        self.n_units = n_units
        self.activation = activation

        layer_units = [np.prod(input_dim)]
        layer_units += [n_units for i in range(n_layers)]
        layer_units += [10]

        layers = []
        for i in range(len(layer_units)-1):
            layers.append(layer_type(layer_units[i],
                                     layer_units[i+1],
                                     period, key_pick, learn_key))
        self.layers = nn.ModuleList(layers)


class RealHashNet(HashNet):
    def forward(self, x, time):
        preactivations = []
        r = x
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                r = self.activation(r)
            r = layer(r, time)
            preactivations.append(r)

        return r, None, preactivations


class ComplexHashNet(HashNet):
    def forward(self, x, time):
        preactivations = []
        r_a, r_b = x, torch.zeros_like(x)
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                r_a = self.activation(r_a)
                r_b = self.activation(r_b)
            r_a, r_b = layer(r_a, r_b, time)
            preactivations.append(r_a)

        return r_a, r_b, preactivations


class ResidualHashNet(HashNet):
    def forward(self, x, time):
        preactivations = []
        r_a, r_b = x, torch.zeros_like(x_a)
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                r_a = self.activation(r_a) + t_a
                r_b = self.activation(r_b) + t_b
            t_a, t_b = layer(r_a, r_b, time)
            preactivations.append(r_a)

        return t_a, t_b, preactivations


class LocalHashNet(nn.Module):
    def __init__(self, input_dim, n_units, activation, n_layers, period, key_pick, learn_key, layer_keys):
        super(LocalHashNet, self).__init__()
        self.n_units = n_units
        self.activation = activation

        layer_units = [np.prod(input_dim)]
        layer_units += [n_units for i in range(n_layers)]
        layer_units += [10]

        layers = []
        for i in range(len(layer_units)-1):
            layers.append(cl.HashLinear(layer_units[i], layer_units[i+1], layer_keys[i], key_pick, learn_key))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, time):
        preactivations = []
        r_a, r_b = x, torch.zeros_like(x)
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                r_a = self.activation(r_a)
                r_b = self.activation(r_b)
            r_a, r_b = layer(r_a, r_b, time)
            preactivations.append(r_a)

        return r_a, r_b, preactivations


class LocalPhaseHashNet(nn.Module):
    def __init__(self, input_dim, n_units, activation, n_layers, period, key_pick, learn_key, layer_keys):
        super(LocalPhaseHashNet, self).__init__()
        self.n_units = n_units
        self.activation = activation
        self.time_slow = time_slow
        self.cheat_period = cheat_period

        layer_units = [np.prod(input_dim)]
        layer_units += [n_units for i in range(n_layers)]
        layer_units += [10]

        layers = []
        biases =[]
        for i in range(len(layer_units)-1):
            layers.append(cl.HashTransform(layer_units[i], layer_units[i+1], layer_keys[i], key_pick, learn_key))
            biases.append(nn.Parameter(torch.zeros(layer_units[i+1])))
        self.layers = nn.ModuleList(layers)
        self.biases = nn.ParameterList(biases)

    def forward(self, x, time):
        preactivations = []
        r_a, r_b = x, torch.zeros_like(x)
        for layer_i, layer in enumerate(self.layers):
            if layer_i > 0:
                mag = torch.norm(torch.stack((r_a, r_b)), p=2, dim=0)
                out_mag = self.activation(mag + self.biases[layer_i-1])
                r_a *= out_mag/(mag + 1e-5)
                r_b *= out_mag/(mag + 1e-5)
            r_a, r_b = layer(r_a, r_b, time)
            preactivations.append(r_a)

        mag = torch.norm(torch.stack((r_a, r_b)), p=2, dim=0)
        out_mag = mag + self.biases[-1]
        r_a *= out_mag/(mag + 1e-5)
        r_b *= out_mag/(mag + 1e-5)
        return r_a, r_b, preactivations


class HashConvNet(nn.Module):
    def __init__(self, input_dim, n_units, activation, period, key_pick):
        super(HashConvNet, self).__init__()
        if input_dim[1] != input_dim[2]:
            raise Exception('Only square inputs are supported')
        self.n_units = n_units
        self.activation = activation

        linear_layer = input_dim[1]
        self.lc1 = cl.HashConvSpCh(input_dim[0], n_units, 5, period)
        linear_layer -= 4
        linear_layer //= 2
        self.lc2 = cl.HashConvSpCh(n_units, n_units*2, 3, period)
        linear_layer -= 2
        linear_layer //= 2
        self.lc3 = cl.HashConvSpCh(n_units*2, n_units*4, 3, period)
        linear_layer -= 2
        self.lfc = cl.HashLinear(n_units*4*linear_layer*linear_layer, 10, period, key_pick)

    def forward(self, x_a, time):
        preactivations = []
        r_a, r_b = self.lc1(x_a, torch.zeros_like(x_a), time)
        r_a = F.max_pool2d(r_a, kernel_size = (2,2))
        r_b = F.max_pool2d(r_b, kernel_size = (2,2))
        #preactivations.append(r_a)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)

        r_a, r_b = self.lc2(r_a, r_b, time)
        r_a = F.max_pool2d(r_a, kernel_size = (2,2))
        r_b = F.max_pool2d(r_b, kernel_size = (2,2))
        preactivations.append(r_a)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)

        r_a, r_b = self.lc3(r_a, r_b, time)
        preactivations.append(r_a)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)

        r_a = r_a.view(r_a.shape[0], -1)
        r_b = r_b.view(r_b.shape[0], -1)

        r_a, r_b = self.lfc(r_a, r_b, time)
        
        return r_a, r_b, preactivations


class CifarHashConvNet(nn.Module):
    def __init__(self, input_dim, n_units, n_layers, activation, period, key_pick):
        super(CifarHashConvNet, self).__init__()
        if input_dim[1] != input_dim[2]:
            raise Exception('Only square inputs are supported')
        self.n_units = n_units
        self.activation = activation
        self.time_slow = time_slow
        self.cheat_period = cheat_period

        linear_layer = input_dim[1]
        self.input_layer = cl.HashConvSpCh(input_dim[0], n_units, 5, period)
        linear_layer -= 4
        hidden_layers = []
        for _ in range(n_layers-1):
            hidden_layers.append(cl.HashConvSpCh(n_units, n_units, 5, period))
            linear_layer -= 4
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.lfc1 = cl.HashLinear(n_units*linear_layer*linear_layer, 1000, period, key_pick)
        self.lfc2 = cl.HashLinear(1000, 10, period, key_pick)

    def forward(self, x_a, time):
        preactivations = []
        r_a, r_b = self.input_layer(x_a, torch.zeros_like(x_a), time)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        for layer in self.hidden_layers:
            r_a, r_b = layer(r_a, r_b, time)
            preactivations.append(r_a)
            r_a = self.activation(r_a)
            r_b = self.activation(r_b)
            
        r_a = r_a.view(r_a.shape[0], -1)
        r_b = r_b.view(r_b.shape[0], -1)

        r_a, r_b = self.lfc1(r_a, r_b, time)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        r_a, r_b = self.lfc2(r_a, r_b, time)
        
        return r_a, r_b, preactivations


class CifarHashConvNet_(nn.Module):
    def __init__(self, input_dim, n_units, n_layers, activation, period, key_pick):
        super(CifarHashConvNet_, self).__init__()
        if input_dim[1] != input_dim[2]:
            raise Exception('Only square inputs are supported')
        self.n_units = n_units
        self.activation = activation

        linear_layer = input_dim[1]
        self.input_layer = cl.ComplexConv(input_dim[0], n_units, (5,5))
        linear_layer -= 4
        hidden_layers = []
        for _ in range(n_layers-1):
            hidden_layers.append(cl.HashConvSpCh(n_units, n_units, 5, period))
            linear_layer -= 4
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.lfc1 = cl.HashLinear(n_units, 256, period, key_pick=key_pick)
        self.lfc2 = cl.HashLinear(256, 10, period, key_pick=key_pick)

    def forward(self, x_a, time):
        preactivations = []
        r_a, r_b = self.input_layer(x_a, torch.zeros_like(x_a), time)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        for layer in self.hidden_layers:
            r_a, r_b = layer(r_a, r_b, time)
            preactivations.append(r_a)
            r_a = self.activation(r_a)
            r_b = self.activation(r_b)
        r_a = r_a.view(r_a.shape[0], r_a.shape[1], r_a.shape[2]*r_a.shape[3])
        r_b = r_b.view(r_b.shape[0], r_b.shape[1], r_b.shape[2]*r_b.shape[3])
        r_a, r_b = r_a.mean(2), r_b.mean(2)

        r_a, r_b = self.lfc1(r_a, r_b, time)
        r_a = self.activation(r_a)
        r_b = self.activation(r_b)
        r_a, r_b = self.lfc2(r_a, r_b, time)
        
        return r_a, r_b, preactivations


class StandardFewShotConvNet(nn.Module):
    def __init__(self):
        super(StandardFewShotConvNet, self).__init__()
        n_hidden = 64
        self.conv_0 = nn.Conv2d(3, n_hidden, (3,3), padding=1)
        self.bn_0 = nn.BatchNorm2d(n_hidden, affine=False, track_running_stats=False) 
        self.conv_1 = nn.Conv2d(n_hidden, n_hidden, (3,3), padding=1)
        self.bn_1 = nn.BatchNorm2d(n_hidden, affine=False, track_running_stats=False) 
        self.conv_2 = nn.Conv2d(n_hidden, n_hidden, (3,3), padding=1)
        self.bn_2 = nn.BatchNorm2d(n_hidden, affine=False, track_running_stats=False) 
        self.conv_3 = nn.Conv2d(n_hidden, n_hidden, (3,3), padding=1)
        self.bn_3 = nn.BatchNorm2d(n_hidden, affine=False, track_running_stats=False) 

        self.layers = [self.conv_0,
                       self.conv_1,
                       self.conv_2,
                       self.conv_3]

        self.norms = [self.bn_0,
                      self.bn_1,
                      self.bn_2,
                      self.bn_3]

    def get_coeffs(self, clone):
        return []
    
    def set_coeffs(self, coeffs):
        pass

    def reset_coeffs(self, layer_ids):
        pass

    def forward(self, x, time):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.max_pool2d(x, (2,2), 2, 0)
            x = torch.relu(x)

        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])

        return x

