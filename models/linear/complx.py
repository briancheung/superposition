import math
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.complex_var import (cmul,
                                from_polar,
                                to_polar,
                                ComplexVar)


class RealLinear(nn.Module):
    '''Complex layer that throws away imaginary part'''
    def __init__(self, n_in, n_out):
        super(RealLinear, self).__init__()
        a = torch.empty(n_in, n_out)
        a = nn.init.xavier_normal_(a)
        b = torch.zeros(n_in, n_out)
        self.cv = ComplexVar(a, b, polar_init=True)
        self.w = nn.Parameter(self.cv.s_r)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x_a, x_b):
        x_b = x_b*0
        w_a = self.w[0]
        w_b = self.w[1]
        r_a = torch.mm(x_a, w_a) - torch.mm(x_b, w_b)
        r_b = torch.mm(x_b, w_a) + torch.mm(x_a, w_b)
        return r_a + self.bias, r_b


class ComplexLinear(nn.Module):
    '''Complex layer'''
    def __init__(self, n_in, n_out):
        super(ComplexLinear, self).__init__()
        a = torch.empty(n_in, n_out)
        a = nn.init.xavier_normal_(a)
        b = torch.Tensor(n_in, n_out).uniform_(-np.pi, np.pi) 
        self.cv = ComplexVar(a, b, polar_init=True)
        self.w = nn.Parameter(self.cv.s_r)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x_a, x_b):
        w_a = self.w[0]
        w_b = self.w[1]
        r_a = torch.mm(x_a, w_a) - torch.mm(x_b, w_b)
        r_b = torch.mm(x_b, w_a) + torch.mm(x_a, w_b)
        return r_a + self.bias, r_b


class SwapComplexLinear(nn.Module):
    '''Complex layer with unique weights for every context'''
    def __init__(self, n_in, n_out, period):
        super(SwapComplexLinear, self).__init__()
        w_r = nn.init.xavier_normal_(torch.empty(period, n_in, n_out))
        w_phi = torch.Tensor(period, n_in, n_out).uniform_(-np.pi, np.pi)
        self.w = nn.Parameter(torch.stack(from_polar(w_r, w_phi)))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x_a, x_b, time):
        net_time = time % self.w.shape[1]
        w_a = self.w[0, net_time]
        w_b = self.w[1, net_time]
        r_a = torch.mm(x_a, w_a) - torch.mm(x_b, w_b)
        r_b = torch.mm(x_b, w_a) + torch.mm(x_a, w_b)
        return r_a + self.bias, r_b


def pick_key(pick_method, keys, time):
    if pick_method == 'hash':
        net_time = int(time) % keys.shape[1]
        o = keys[:, net_time]
    elif pick_method == 'local_mix':
        center_time = int(time)
        b_time = (center_time-1) % keys.shape[1]
        m_time = center_time % keys.shape[1]
        e_time = (center_time+1) % keys.shape[1]
        o_r, o_phi = to_polar(keys)
        o = torch.stack(from_polar(o_r.mean(0), (o_phi[b_time] + o_phi[m_time] + o_phi[e_time])/3))
    elif pick_method == 'local_mult':
        center_time = int(time)
        b_time = (center_time-1) % keys.shape[1]
        m_time = center_time % keys.shape[1]
        e_time = (center_time+1) % keys.shape[1]
        o = cmul(cmul(keys[:, b_time], keys[:, m_time]), keys[:, e_time])
    elif pick_method == 'temp_mix':
        net_time = torch.tensor([int(time) % keys.shape[1]])
        key_logit = torch.zeros(1, keys.shape[1]).scatter_(1, net_time.unsqueeze(1), 1./(time/1000.+1e-5))
        key_prob = F.softmax(key_logit, 1).cuda()
        o_r, o_phi = to_polar(keys)
        o_r_pick = torch.matmul(key_prob.squeeze(), o_r)
        o_phi_pick = torch.matmul(key_prob.squeeze(), o_phi)
        o = torch.stack(from_polar(o_r_pick, o_phi_pick))
    elif pick_method == 'random':
        net_time = np.random.randint(keys.shape[1])
        o = keys[:, net_time]
    elif pick_method == 'cosine':
        omega = (int(time) % keys.shape[1])*np.pi/keys.shape[1]
        mix = (torch.cos(torch.tensor(omega).cuda()) + 1)/2.
        o = (mix*keys[:, 0]) + ((1.-mix)*keys[:, 1])
    elif pick_method == 'triangle_multiply':
        net_time = int(time) % keys.shape[1]
        o_r, o_phi = to_polar(keys)
        o = torch.stack(from_polar(o_r.mean(0), o_phi[:net_time+1].sum(0)))
    elif pick_method == 'one_power':
        net_time = time % keys.shape[1]
        o_r, o_phi = to_polar(keys)
        o = torch.stack(from_polar(o_r[0], net_time*o_phi[0]))
    else:
        raise NotImplementedError

    return o


class HashLinear(nn.Module):
    '''Complex layer with complex diagonal contexts'''
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(HashLinear, self).__init__()
        self.key_pick = key_pick
        w_r = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        w_phi = torch.Tensor(n_in, n_out).uniform_(-np.pi, np.pi)
        o_r = torch.ones(period, n_in)
        o_phi = torch.Tensor(period, n_in).uniform_(-np.pi, np.pi)

        self.w = nn.Parameter(torch.stack(from_polar(w_r, w_phi)))
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(torch.stack(from_polar(o_r, o_phi)))
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x_a, x_b, time):
        o = pick_key(self.key_pick, self.o, time)
        o_a = o[0].unsqueeze(0)
        o_b = o[1].unsqueeze(0)
        m_a = x_a*o_a - x_b*o_b
        m_b = x_b*o_a + x_a*o_b

        w_a = self.w[0]
        w_b = self.w[1]
        r_a = torch.mm(m_a, w_a) - torch.mm(m_b, w_b)
        r_b = torch.mm(m_b, w_a) + torch.mm(m_a, w_b)
        return r_a + self.bias, r_b


class HashTransform(nn.Module):
    '''Complex layer with complex diagonal contexts without bias'''
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(HashTransform, self).__init__()
        self.key_pick = key_pick
        w_r = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        w_phi = torch.Tensor(n_in, n_out).uniform_(-np.pi, np.pi)
        o_r = torch.ones(period, n_in)
        o_phi = torch.Tensor(period, n_in).uniform_(-np.pi, np.pi)

        self.w = nn.Parameter(torch.stack(from_polar(w_r, w_phi)))
        self.o = nn.Parameter(torch.stack(from_polar(o_r, o_phi)))
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x_a, x_b, time):
        o = pick_key(self.key_pick, self.o, time)
        o_a = o[0].unsqueeze(0)
        o_b = o[1].unsqueeze(0)
        m_a = x_a*o_a - x_b*o_b
        m_b = x_b*o_a + x_a*o_b

        w_a = self.w[0]
        w_b = self.w[1]
        r_a = torch.mm(m_a, w_a) - torch.mm(m_b, w_b)
        r_b = torch.mm(m_b, w_a) + torch.mm(m_a, w_b)
        return r_a, r_b


class FourierLinear(nn.Module):
    '''Complex layer with fourier diagonal contexts'''
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(FourierLinear, self).__init__()
        self.key_pick = key_pick
        w_r = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        w_phi = torch.Tensor(n_in, n_out).uniform_(-np.pi, np.pi)
        o_r = torch.ones(period, n_in)
        o_phi = torch.Tensor(period, n_in)
        #o_phi = torch.Tensor(period, n_in).uniform_(-np.pi, np.pi)
        for i in range(n_in):
            o_phi[:,i] = (2*np.pi*(i+1))/period

        self.w = nn.Parameter(torch.stack(from_polar(w_r, w_phi)))
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(torch.stack(from_polar(o_r, o_phi)))
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x_a, x_b, time):
        net_time = time % self.o.shape[1]
        o_r, o_phi = to_polar(self.o)
        o = torch.stack(from_polar(o_r[0], net_time*o_phi[0]))

        o_a = o[0].unsqueeze(0)
        o_b = o[1].unsqueeze(0)
        m_a = x_a*o_a - x_b*o_b
        m_b = x_b*o_a + x_a*o_b

        w_a = self.w[0]
        w_b = self.w[1]
        r_a = torch.mm(m_a, w_a) - torch.mm(m_b, w_b)
        r_b = torch.mm(m_b, w_a) + torch.mm(m_a, w_b)
        return r_a + self.bias, r_b


class ComplexConv(nn.Module):
    def __init__(self, n_chin, n_chout, kernel_size, padding=0, init_mult=1.):
        super(ComplexConv, self).__init__()
        n = n_chin
        for k in kernel_size:
            n *= k
        stdv = 1./math.sqrt(n)
        w_r = torch.Tensor(n_chout, n_chin, *kernel_size).uniform_(-stdv, stdv)
        w_phi = torch.Tensor(n_chout, n_chin, *kernel_size).uniform_(-np.pi, np.pi)

        self.w = nn.Parameter(torch.stack(from_polar(init_mult*w_r, w_phi)))
        self.bias = nn.Parameter(torch.Tensor(n_chout).uniform_(-stdv, stdv))
        self.padding = padding
        
    def forward(self, x_a, x_b, time):
        w_a = self.w[0]
        w_b = self.w[1]
        r_a = F.conv2d(x_a, w_a,
                       bias=self.bias,
                       padding=self.padding) - F.conv2d(x_b, w_b, padding=self.padding)
        r_b = F.conv2d(x_b, w_a, padding=self.padding) + F.conv2d(x_a, w_b, padding=self.padding)
        return r_a, r_b


class HashConv(nn.Module):
    def __init__(self, n_chin, n_chout, kernel_size, period, key_pick='hash', learn_key=True):
        super(HashConv, self).__init__()
        self.key_pick = key_pick
        n = n_chin
        for k in kernel_size:
            n *= k
        stdv = 1./math.sqrt(n)
        w_r = torch.Tensor(n_chout, n_chin, *kernel_size).uniform_(-stdv, stdv)
        w_phi = torch.Tensor(n_chout, n_chin, *kernel_size).uniform_(-np.pi, np.pi)
        o_r = torch.ones(period, n_chout)
        o_phi = torch.Tensor(period, n_chout).uniform_(-np.pi, np.pi)

        self.w = nn.Parameter(torch.stack(from_polar(w_r, w_phi)))
        self.bias = nn.Parameter(torch.Tensor(n_chout).uniform_(-stdv, stdv))
        self.o = nn.Parameter(torch.stack(from_polar(o_r, o_phi)))
        
    def forward(self, x_a, x_b, time):
        w_a = self.w[0]
        w_b = self.w[1]
        out_a, out_b = 0., 0.
        for i in range(self.o.shape[1]):
            o_a = self.o[0,i].view(1, -1, 1, 1) 
            o_b = self.o[1,i].view(1, -1, 1, 1)
            r_a = F.conv2d(x_a, w_a, self.bias) - F.conv2d(x_b, w_b)
            r_b = F.conv2d(x_b, w_a) + F.conv2d(x_a, w_b)
            m_a = r_a*o_a - r_b*o_b
            m_b = r_b*o_a + r_a*o_b
            
            out_a += m_a
            out_b += m_b
            print('FM:', i)
        return r_a, r_b


class HashConvSpCh(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, period, key_pick='hash', learn_key=False):
        super(HashConvSpCh, self).__init__()
        self.key_pick = key_pick
        self.period = period
        n = n_in

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        for k in kernel_size:
            n *= k
        stdv = 1./math.sqrt(n)
        w_r = torch.empty(n_out, n_in, kernel_size[0], kernel_size[1]).uniform_(-stdv,stdv)
        w_phi = torch.Tensor(*w_r.shape).uniform_(-np.pi, np.pi)
        o_r = torch.ones(*((period,1) + w_r.shape[1:]))
        o_phi = torch.Tensor(*o_r.shape).uniform_(-np.pi, np.pi)

        self.w = nn.Parameter(torch.stack(from_polar(w_r, w_phi)))
        self.bias = nn.Parameter(torch.torch.empty(n_out,).uniform_(-stdv,stdv))

        o = torch.stack(from_polar(o_r, o_phi))
        self.o = nn.Parameter(o, requires_grad=learn_key)

    def forward(self, x_a, x_b, time):
        o = pick_key(self.key_pick, self.o, time)
        o_a = o[0]
        o_b = o[1]
        # Cheaper to contextualize the weights instead of the input for convolution
        w_a = self.w[0]*o_a - self.w[1]*o_b
        w_b = self.w[1]*o_a + self.w[0]*o_b

        r_a = F.conv2d(x_a, w_a, self.bias) - F.conv2d(x_b, w_b)
        r_b = F.conv2d(x_b, w_a) + F.conv2d(x_a, w_b)
        return r_a, r_b
