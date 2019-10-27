import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import datasets as ds 
from models.linear import complx, real
from models import (baseline,
                    contextnet as ctxnet,
                    contextlayer as ctxlayer,
                    baseline,
                    resnet)


def get_dataset(name, period, batch_size, train, kwargs):
    if name == 'permuting_mnist':
        data_loader = ds.PermutingMNIST(period,
                                        batch_size,
                                        4321,
                                        train=train,
                                        kwargs=kwargs)
    elif name == 'permuting_cifar':
        data_loader = ds.PermutingCIFAR(period,
                                        batch_size,
                                        4321,
                                        train=train,
                                        kwargs=kwargs)
    elif name == 'permuting_fashionmnist':
        data_loader = ds.PermutingFashionMNIST(period,
                                               batch_size,
                                               4321,
                                               train=train,
                                               kwargs=kwargs)
    elif name == 'permuting_svhn':
        if train:
            split = 'extra'
        else:
            split = 'test'
        data_loader = ds.PermutingSVHN(period,
                                       batch_size,
                                       4321,
                                       split=split,
                                       kwargs=kwargs)
    elif name == 'rotating_mnist':
        data_loader = ds.RotatingMNIST(period,
                                       batch_size,
                                       train=train,
                                       draw_and_rotate=True,
                                       kwargs=kwargs)
    elif name == 'rotating_fashionmnist':
        data_loader = ds.RotatingFashionMNIST(period,
                                              batch_size,
                                              train=train,
                                              draw_and_rotate=True,
                                              kwargs=kwargs)
    elif name == 'rotating_cifar':
        data_loader = ds.RotatingCIFAR(period,
                                       batch_size,
                                       train=train,
                                       draw_and_rotate=True,
                                       kwargs=kwargs)
    elif name == 'rotating_persistent_mnist':
        data_loader = ds.RotatingMNIST(period,
                                       batch_size,
                                       train=train,
                                       draw_and_rotate=False,
                                       kwargs=kwargs)
    elif name == 'rocating_mnist':
        data_loader = ds.RocatingMNIST(period,
                                       batch_size,
                                       train=train,
                                       draw_and_rotate=True,
                                       kwargs=kwargs)
    elif name == 'incrementing_cifar':
        data_loader = ds.IncrementingCIFAR(period,
                                           batch_size,
                                           train=train,
                                           kwargs=kwargs)
    elif name == 'incrementing_cifar100':
        data_loader = ds.IncrementingCIFAR(period,
                                           batch_size,
                                           train=train,
                                           n_class=5,
                                           use_cifar10=False,
                                           kwargs=kwargs)

    return data_loader


def get_activation(name): 
    if name == 'tanh':
        activation = torch.tanh
    elif name == 'sigmoid':
        activation = torch.sigmoid
    elif name == 'relu':
        activation = torch.relu
    elif name == 'none':
        activation = torch.identity

    return activation


def get_fc_net(name, input_dim, output_dim, activation, args):
    mynet = None
    if name == 'pytorch':
        mynet = baseline.BaselineNet(input_dim,
                                     args.n_units,
                                     activation,
                                     args.n_layers,
                                     nn.Linear)
    elif name == 'real':
        mynet = baseline.SimpleNet(input_dim,
                                   args.n_units,
                                   activation,
                                   args.n_layers,
                                   complx.RealLinear)
    elif name == 'complex':
        mynet = baseline.SimpleNet(input_dim,
                                   args.n_units,
                                   activation,
                                   args.n_layers,
                                   complx.ComplexLinear)
    elif name == 'hash':
        mynet = ctxnet.ComplexHashNet(input_dim,
                                      args.n_units,
                                      activation,
                                      args.n_layers,
                                      args.net_period,
                                      args.key_pick,
                                      args.learn_key,
                                      complx.HashLinear)
    elif name == 'smoothhash':
        mynet = ctxnet.ComplexHashNet(input_dim,
                                      args.n_units,
                                      activation,
                                      args.n_layers,
                                      args.net_period,
                                      args.key_pick,
                                      args.learn_key,
                                      complx.FourierLinear)
    elif name == 'localhash':
        mynet = ctxnet.LocalHashNet(input_dim,
                                    args.n_units,
                                    activation,
                                    args.n_layers,
                                    args.net_period,
                                    args.key_pick,
                                    args.learn_key,
                                    args.l_period)
    elif name == 'localphasehash':
        mynet = ctxnet.LocalPhaseHashNet(input_dim,
                                         args.n_units,
                                         activation,
                                         args.n_layers,
                                         args.net_period,
                                         args.key_pick,
                                         args.learn_key,
                                         args.l_period)
    elif name == 'layersp':
        mynet = ctxlayer.LayerSPNet(input_dim,
                                    args.n_units,
                                    activation,
                                    args.n_layers,
                                    args.learn_key)
    elif name == 'layerspres':
        mynet = ctxlayer.LayerSPResNet(input_dim,
                                       args.n_units,
                                       activation,
                                       args.n_layers,
                                       args.learn_key)
    elif name == 'layerspres_':
        mynet = ctxlayer.LayerSPResNet_(input_dim,
                                        args.n_units,
                                        activation,
                                        args.n_layers,
                                        args.learn_key)
    elif name == 'recurrentl':
        mynet = ctxlayer.RecurrentLayerNet(input_dim,
                                           args.n_units,
                                           activation,
                                           args.n_layers)
    elif name == 'recurrentlres':
        mynet = ctxlayer.RecurrentLayerNet(input_dim,
                                           args.n_units,
                                           activation,
                                           args.n_layers)
    elif name in {'binaryhash',
                  'hadamardhash', 
                  'maskhash',
                  'ternaryhash',
                  'normalhash',
                  'househash',
                  'rotatehash',
                  'rotatepowhash',
                  'routehash',
                  'binroutehash',
                  'multinet'}:
        if name == 'binaryhash':
            layer_type = real.BinaryHashLinear
        elif name == 'hadamardhash':
            layer_type = real.HadamardHashLinear
        elif name == 'maskhash':
            layer_type = real.MaskHashLinear
        elif name == 'ternaryhash':
            layer_type = real.TernaryHashLinear
        elif name == 'normalhash':
            layer_type = real.NormalHashLinear
        elif name == 'househash':
            layer_type = real.HouseholderLinear
        elif name == 'rotatehash':
            layer_type = real.RotateLinear
        elif name == 'rotatepowhash':
            layer_type = real.RotatePowerLinear
        elif name == 'routehash':
            layer_type = real.RouteLinear
        elif name == 'binroutehash':
            layer_type = real.BinRouteLinear
        elif name == 'multinet':
            layer_type = real.MultiHeadLinear
        mynet = ctxnet.RealHashNet(input_dim,
                                   args.n_units,
                                   activation,
                                   args.n_layers,
                                   args.net_period,
                                   args.key_pick,
                                   args.learn_key,
                                   layer_type)
    elif name == 'residualhash':
        mynet = ctxnet.ResidualHashNet(input_dim,
                                       args.n_units,
                                       activation,
                                       args.n_layers,
                                       args.net_period,
                                       args.key_pick)

    return mynet


def get_conv_net(name, input_dim, output_dim, activation, args):
    mynet = None
    if name == 'complexconv':
        mynet = baseline.ComplexConvNet(input_dim,
                                        args.n_units,
                                        activation,
                                        args.n_layers)
    elif name == 'hashconv':
        mynet = ctxnet.HashConvNet(input_dim = input_dim,
                                   n_units = args.n_units,
                                   activation = activation,
                                   period = args.net_period,
                                   key_pick = args.key_pick)
    elif name == 'cifarhashconv':
        mynet = ctxnet.CifarHashConvNet(input_dim = input_dim,
                                        n_units = args.n_units,
                                        n_layers = args.n_layers,
                                        activation = activation,
                                        period = args.net_period,
                                        key_pick = args.key_pick)
    elif name == 'cifarhashconv_':
        mynet = ctxnet.CifarHashConvNet_(input_dim = input_dim,
                                         n_units = args.n_units,
                                         n_layers = args.n_layers,
                                         activation = activation,
                                         period = args.net_period,
                                         key_pick = args.key_pick)
    elif name == 'residualcomplexconv':
        mynet = baseline.ResidualComplexConvNet(input_dim,
                                                args.n_units,
                                                activation,
                                                args.n_layers)
    elif name == 'resnet18':
        mynet = resnet.ResNet18()
    elif name == 'staticbnresnet18':
        mynet = resnet.StaticBNResNet18()
    elif name == 'outhashresnet18':
        mynet = resnet.OutHashResNet18()
    elif name == 'hashresnet18':
        mynet = resnet.HashResNet18(np.prod(output_dim))
    elif name == 'multiresnet18':
        mynet = resnet.MultiHeadResNet18()

    return mynet


def get_optimizer(name, params, args):
    if name == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    elif name == 'adam':
        optimizer = optim.Adam(params, lr=args.lr)
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=args.lr)
    
    return optimizer
