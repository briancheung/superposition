import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def is_power2(num):
    'states if a number is a power of two'
    # From: http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
    # Author: A.Polino
    return num != 0 and ((num & (num - 1)) == 0)


class BinaryHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(BinaryHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        rand_01 = np.random.binomial(p=.5, n=1, size=(n_in, period)).astype(np.float32)
        o = torch.from_numpy(rand_01*2 - 1)

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, time):
        o = self.o[:, int(time)]
        m = x*o
        r = torch.mm(m, self.w)
        return r


class HadamardHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(HadamardHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))

        if is_power2(n_in):
            o = torch.from_numpy(scipy.linalg.hadamard(n_in)[:period].T.astype(np.float32))
        else:
            rand_01 = np.random.binomial(p=.5, n=1, size=(n_in, period)).astype(np.float32)
            o = torch.from_numpy(rand_01*2 - 1)

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, time):
        o = self.o[:, int(time)]
        m = x*o
        r = torch.mm(m, self.w)
        return r


class MaskHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(MaskHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        rand_01 = np.random.binomial(p=.5, n=1, size=(n_in, period)).astype(np.float32)
        o = torch.from_numpy(rand_01)

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, time):
        o = self.o[:, int(time)] 
        m = x*o
        r = torch.mm(m, self.w)
        return r


class TernaryHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(TernaryHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        rand_n101 = np.random.randint(low=-1, high=2, size=(n_in, period)).astype(np.float32)
        o = torch.from_numpy(rand_n101)

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, time):
        o = self.o[:, int(time)] 
        m = x*o
        r = torch.mm(m, self.w)
        return r


class NormalHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(NormalHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        o = torch.from_numpy(np.random.randn(n_in, period).astype(np.float32))

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, time):
        o = self.o[:, int(time)]
        m = x*o
        r = torch.mm(m, self.w)
        return r


class HouseholderLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(HouseholderLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        q_list = []
        for key_i in range(period):
            v = np.random.randn(n_in).astype('float32')
            q_list.append(torch.from_numpy(v))

        vs = torch.stack(q_list)
        self.vs = nn.Parameter(vs)

        if not learn_key:
            self.vs.requires_grad = False

    def forward(self, x, time):
        v = self.vs[int(time)]
        v = v/torch.sqrt(torch.sum(v**2))
        rot = torch.eye(v.shape[0]).cuda() - 2.*torch.ger(v, v)
        m = torch.mm(x, torch.t(rot))
        return torch.mm(m, self.w) + self.bias


class RotateLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(RotateLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        q_list = []
        for key_i in range(period):
            #a = nn.init.xavier_normal_(torch.empty(n_in, n_in))
            #q, r = torch.qr(a)
            q = torch.from_numpy(scipy.stats.ortho_group.rvs(n_in).astype('float32'))
            q_list.append(q)

        rots = torch.stack(q_list)
        self.rots = nn.Parameter(rots)

        if not learn_key:
            self.rots.requires_grad = False

    def forward(self, x, time):
        m = torch.mm(x, self.rots[int(time)])
        return torch.mm(m, self.w) + self.bias


class RouteLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(RouteLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        rots = []
        
        for key_i in range(period):
            row_idx = np.random.permutation(n_in)
            random_route = np.eye(n_in)[row_idx].astype('float32')
            rots.append(torch.from_numpy(random_route))

        rots = torch.stack(rots)
        self.rots = nn.Parameter(rots)
        
        if not learn_key:
            self.rots.requires_grad = False
    
    def forward(self, x, time):
        m = torch.mm(x, self.rots[int(time)])
        return torch.mm(m, self.w) + self.bias


class RotatePowerLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(RotatePowerLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))

        context = np.eye(n_in).astype('float32')
        src_context = scipy.stats.ortho_group.rvs(n_in).astype('float32')
        q_list = []
        for key_i in range(period):
            #a = nn.init.xavier_normal_(torch.empty(n_in, n_in))
            #q, r = torch.qr(a)
            context = np.dot(context, src_context)
            q = torch.from_numpy(context)
            q_list.append(q)

        rots = torch.stack(q_list)
        self.rots = nn.Parameter(rots)

        if not learn_key:
            self.rots.requires_grad = False

    def forward(self, x, time):
        m = torch.mm(x, self.rots[int(time)])
        return torch.mm(m, self.w) + self.bias


class BinRouteLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(BinRouteLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        
        o = np.random.binomial(p=.5, n=1, size = (n_in, period)).astype('float32')*2 - 1
        self.o = nn.Parameter(torch.from_numpy(o))
        self.routes = [np.random.permutation(n_in) for key_i in range(period)]

        if not learn_key:
            self.o.requires_grad = False
    
    def forward(self, x, time):
        m = x[:,self.routes[int(time)]]*self.o[:,int(time)]
        return torch.mm(m, self.w) + self.bias


class MultiHeadLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick=None, learn_key=False):
        super(MultiHeadLinear, self).__init__()
        w_stack = []
        for i in range(period):
            w_stack.append(nn.init.xavier_normal_(torch.empty(n_in, n_out)))
        w = torch.stack(w_stack, dim=2) 
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out, period))

    def forward(self, x, time):
        return torch.mm(x, self.w[:,:,int(time)]) + self.bias[:,int(time)]


class HashConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, period, 
                 stride=1, padding=0, bias=True,
                 key_pick='hash', learn_key=True):
        super(HashConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        w = torch.zeros(self.out_channels, self.in_channels, *self.kernel_size)
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        self.w = nn.Parameter(w)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        o_dim = self.in_channels*self.kernel_size[0]*self.kernel_size[1]
        # TODO(briancheung): The line below will cause problems when saving a model
        o = torch.from_numpy( np.random.binomial( p=.5, n=1, size = (o_dim, period) ).astype(np.float32) * 2 - 1 )
        self.o = nn.Parameter(o, requires_grad=False)

    def forward(self, x, time):
        net_time = time % self.o.shape[1]
        o = self.o[:, net_time].view(1,
                                     self.in_channels,
                                     self.kernel_size[0],
                                     self.kernel_size[1])
        return F.conv2d(x, self.w*o, self.bias, stride=self.stride, padding=self.padding)
