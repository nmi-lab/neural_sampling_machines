# =========================================================================
# Implementation of Neural Sampling Machine module
# Copyright (C) <2019>  Georgios Detorakis (gdetor@protonmail.com)
#                       Emre Neftci (eneftci@uci.edu)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =========================================================================
import torch as tr
import numpy as np
from torch import nn
from torch.nn import functional as F


eps = np.finfo(float).eps


def isnan(x):
    """
        Checks if the input is a nan
    """
    return bool((x != x).sum())


class Sign_(tr.autograd.Function):
    """
        Signum function (non-linearity) class
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).type(x.dtype)*2-1

    @staticmethod
    def backward(ctx, grad_output):
        aux, = ctx.saved_tensors
        # grad_input = (1 - tr.tanh(aux)**2) * grad_output
        # grad_input = tr.ones_like(grad_output)
        # return grad_input
        return grad_output


def binconcrete(alpha, temperature=1):
    """
        Implements the BinConcrete distribution (Gumbel)
    """
    U = tr.zeros_like(alpha).uniform_(0, 1)
    L = tr.log(U + eps) - tr.log(1 - U + eps)   # Logistic
    X = tr.sigmoid((L + tr.log(alpha + eps)) / (temperature + eps))
    return X


class Lambda_(nn.Module):
    """
        Lambda input layer (not in use)
    """
    def __init__(self, lambd):
        super(Lambda_, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class GaussianNoise(nn.Module):
    """
        Implements a class for Gaussian Noise layer (normalizer)
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = tr.tensor(0, dtype=tr.float32).cuda()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (self.sigma * x.detach()
                     if self.is_relative_detach else self.sigma * x)
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


class NSMLinear_(nn.Linear):
    """
        Linear Neural Sampling Machine (NSM) Layer
    """
    nsm_layer = True

    def __init__(self, in_features, out_features, bias=True, prob=.5, sigma=1,
                 noise='bernoulli', use_cuda=True):
        super(NSMLinear_, self).__init__(in_features, out_features, bias)
        self.use_cuda = use_cuda
        self.prob = prob
        self.sigma = sigma
        self.noise = noise
        self.sign = Sign_().apply
        self.ap = np.sqrt(2 * prob * (1 - prob))
        self.b = tr.nn.Parameter(tr.ones([self.out_features]))
        self.register_parameter('magnitude', self.b)

    def cdf(self, x):
        wn = self.weight
        w_ = tr.div(wn.permute(1, 0),
                    tr.sqrt(tr.sum(wn**2, dim=1))).permute(1, 0)
        P = self.b * F.linear(x, w_, bias=None)

        if self.bias is not None:
            P += self.bias / self.ap / w_.norm(p=2, dim=1)
        return 0.5 * (1 + tr.erf(P)), P, wn

    def forward(self, x):
        z = (self.sign(x) - x).detach() + x
        # z = self.sign(x)
        phi, P, W = self.cdf(z)

        xu = tr.empty_like(z)
        if self.noise == 'bernoulli':
            xu.bernoulli_(self.prob)
            a = self.b * self.ap - self.prob
        else:
            xu.normal_(1, np.sqrt(self.sigma))
            a = self.b * np.sqrt(2*self.sigma**2) - 1

        u = (F.linear(xu * z, W, bias=None) + a * F.linear(z, W, bias=None))

        if self.bias is not None:
            u += self.bias

        # y = (self.sign(u) - (2 * phi - 1)).detach() + (2 * phi - 1)
        # y = self.sign(u)
        y = 2 * binconcrete(phi) - 1
        return y, u, phi


class dNSMLinear_(nn.Linear):
    """
        Deterministic (no multiplicative noise) linear Neural Sampling Machine
        (NSM) Layer
    """
    nsm_layer = True

    def __init__(self, in_features, out_features, bias=True, prob=.5, sigma=1,
                 noise='bernoulli', use_cuda=True):
        super(dNSMLinear_, self).__init__(in_features, out_features, bias)
        self.use_cuda = use_cuda
        self.prob = prob
        self.sigma = sigma
        self.noise = noise
        self.sign = Sign_().apply
        self.ap = np.sqrt(2 * prob * (1 - prob))
        self.b = tr.nn.Parameter(tr.ones([self.out_features]))
        self.register_parameter('magnitude', self.b)

    def cdf(self, x):
        wn = self.weight
        w_ = tr.div(wn.permute(1, 0),
                    tr.sqrt(tr.sum(wn**2, dim=1))).permute(1, 0)
        P = self.b * F.linear(x, w_, bias=None)

        if self.bias is not None:
            P += self.bias / self.ap / w_.norm(p=2, dim=1)
        return 0.5 * (1 + tr.erf(P)), P, wn

    def forward(self, x):
        z = (self.sign(x) - x).detach() + x
        phi, P, W = self.cdf(z)

        u = (F.linear(z, W, bias=None) + F.linear(z, W, bias=None))

        if self.bias is not None:
            u += self.bias

        y = (self.sign(u) - (2 * phi - 1)).detach() + (2 * phi - 1)
        return y, u, phi


class NSMConv2d_(nn.Conv2d):
    """
        Convolutional Neural Sampling Machine (NSM) Layer
    """
    nsm_layer = True

    def __init__(self, in_channels, out_channels, input_shape=(1, 1),
                 kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True, prob=.5, sigma=1,
                 noise='bernoulli', use_cuda=True):
        super(NSMConv2d_, self).__init__(in_channels, out_channels,
                                         kernel_size, stride, padding,
                                         dilation, groups, bias)
        if len(input_shape) != 2:
            print("Update input_shape to provide width and height only!")
            raise

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_cuda = use_cuda

        self.prob = prob
        self.sigma = sigma
        self.noise = noise
        self.sign = Sign_().apply
        self.ap = np.sqrt(2 * prob * (1 - prob))

        cout, hout, wout = self.conv_shape_(input_shape[0],
                                            input_shape[1])

        self.b = tr.nn.Parameter(tr.ones([cout, hout, wout]))
        self.register_parameter('magnitude', self.b)

    def cdf(self, x):
        wn = self.weight
        wnn = wn.view(wn.shape[0], -1)
        w_ = tr.div(wnn.contiguous().permute(1, 0),
                    tr.sqrt(tr.sum(wnn**2, dim=1))).contiguous().permute(1, 0).contiguous().reshape(wn.shape)

        P = self.b * F.conv2d(x, w_, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)

        # w_norm = tr.sqrt(tr.sum(wnn**2, dim=1))
        # w_norm = wnn.norm(p=2, dim=1)
        if self.bias is not None:
            m, n, p, q = self.weight.size()
            w_norm = wn.view(m, n*p*q).norm(p=2, dim=1)
            P = ((P.permute(0, 3, 2, 1) +
                 (self.bias / self.ap / w_norm)).permute(0, 3, 2, 1))
        return 0.5 * (1 + tr.erf(P)), P, wn

    def forward(self, x):
        z = (self.sign(x) - x).detach() + x
        # z = self.sign(x)
        phi, P, W = self.cdf(z)

        xu = tr.empty_like(z)
        if self.noise == 'bernoulli':
            xu.bernoulli_(self.prob)
            a = self.b * self.ap - self.prob
        else:
            xu.normal_(1, np.sqrt(self.sigma))
            a = self.b * np.sqrt(2*self.sigma**2) - 1

        u = (F.conv2d(xu * z, W, bias=None, stride=self.stride,
                      padding=self.padding, dilation=self.dilation,
                      groups=self.groups)
             + a * F.conv2d(z, W, bias=None, stride=self.stride,
                            padding=self.padding, dilation=self.dilation,
                            groups=self.groups))

        if self.bias is not None:
            u = (u.permute(0, 3, 2, 1) + self.bias).permute(0, 3, 2, 1)

        # y = (self.sign(u) - (2 * phi - 1)).detach() + (2 * phi - 1)
        # y = self.sign(u)
        y = 2 * binconcrete(phi) - 1
        return y, u, phi

    def conv_shape_(self, Hin, Win):
        Cout = self.out_channels
        tmp = (1 + (Hin + 2*self.padding[0] - self.dilation[0] *
               (self.kernel_size[0] - 1) - 1) / self.stride[0])
        Hout = int(np.floor(tmp))
        tmp = (1 + (Win + 2 * self.padding[1] - self.dilation[1] *
               (self.kernel_size[1] - 1) - 1) / self.stride[1])
        Wout = int(np.floor(tmp))
        return Cout, Hout, Wout


class dNSMConv2d_(nn.Conv2d):
    """
        Deterministic (no multiplicative noise) Convolutional Neural
        Sampling Machine (NSM) Layer
    """
    nsm_layer = True

    def __init__(self, in_channels, out_channels, input_shape=(1, 1),
                 kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True, prob=.5, sigma=1,
                 noise='bernoulli', use_cuda=True):
        super(dNSMConv2d_, self).__init__(in_channels, out_channels,
                                          kernel_size, stride, padding,
                                          dilation, groups, bias)
        if len(input_shape) != 2:
            print("Update input_shape to provide width and height only!")
            raise

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_cuda = use_cuda

        self.prob = prob
        self.sigma = sigma
        self.noise = noise
        self.sign = Sign_().apply
        self.ap = np.sqrt(2 * prob * (1 - prob))

        cout, hout, wout = self.conv_shape_(input_shape[0],
                                            input_shape[1])

        self.b = tr.nn.Parameter(tr.ones([cout, hout, wout]))
        self.register_parameter('magnitude', self.b)

    def cdf(self, x):
        wn = self.weight
        wnn = wn.view(wn.shape[0], -1)
        w_ = tr.div(wnn.contiguous().permute(1, 0),
                    tr.sqrt(tr.sum(wnn**2, dim=1))).contiguous().permute(1, 0).contiguous().reshape(wn.shape)

        P = self.b * F.conv2d(x, w_, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)

        if self.bias is not None:
            m, n, p, q = self.weight.size()
            w_norm = wn.view(m, n*p*q).norm(p=2, dim=1)
            P = ((P.permute(0, 3, 2, 1) +
                 (self.bias / self.ap / w_norm)).permute(0, 3, 2, 1))
        return 0.5 * (1 + tr.erf(P)), P, wn

    def forward(self, x):
        z = (self.sign(x) - x).detach() + x
        phi, P, W = self.cdf(z)

        u = (F.conv2d(z, W, bias=None, stride=self.stride,
                      padding=self.padding, dilation=self.dilation,
                      groups=self.groups)
             + F.conv2d(z, W, bias=None, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        groups=self.groups))

        if self.bias is not None:
            u = (u.permute(0, 3, 2, 1) + self.bias).permute(0, 3, 2, 1)

        y = (self.sign(u) - (2 * phi - 1)).detach() + (2 * phi - 1)
        return y, u, phi

    def conv_shape_(self, Hin, Win):
        Cout = self.out_channels
        tmp = (1 + (Hin + 2*self.padding[0] - self.dilation[0] *
               (self.kernel_size[0] - 1) - 1) / self.stride[0])
        Hout = int(np.floor(tmp))
        tmp = (1 + (Win + 2 * self.padding[1] - self.dilation[1] *
               (self.kernel_size[1] - 1) - 1) / self.stride[1])
        Wout = int(np.floor(tmp))
        return Cout, Hout, Wout
