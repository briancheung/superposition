import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def from_polar(r, phi):
    a = r*torch.cos(phi)
    b = r*torch.sin(phi)
    return a, b


def to_polar(v):
    r = torch.norm(v, p=2, dim=0)
    phi = torch.atan2(v[1], v[0])
    return r, phi


def cmul(k, v):
    a = k[0]*v[0] - k[1]*v[1]
    b = k[1]*v[0] + k[0]*v[1]
    return torch.stack([a, b])


def cdiv(k, v):
    a = k[0]*v[0] + k[1]*v[1]
    b = k[1]*v[0] - k[0]*v[1]
    return torch.stack([a, b])/(v[0]**2 + v[1]**2)


class ComplexVar(object):
    def __init__(self, a, b, polar_init=False):
        self.s_r = torch.FloatTensor(*((2,) + a.shape))
        if polar_init:
            a, b = from_polar(a, b)

        self.s_r[0] = a
        self.s_r[1] = b

    def to_polar(self):
        return to_polar(self.s_r) 


class QuaternionVar(object):
    def __init__(self, a, b, c, d):
        self.s_r[0] = a
        self.s_r[1] = b
        self.s_r[2] = c
        self.s_r[3] = d
