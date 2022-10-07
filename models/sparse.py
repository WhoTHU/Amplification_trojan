import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def act(x, types='soft_single', act_args=0.1):
    return {
        'relu':lambda x, act_args: F.relu(x),
        'soft_single':lambda x, act_args: (x - act_args).max(torch.zeros_like(x)),
        'soft_double':lambda x, act_args: (x - act_args).max(torch.zeros_like(x)) + (x + act_args).min(torch.zeros_like(x)),
        'hard':lambda x, act_args: x * (x > act_args).type_as(x)
    }[types](x, act_args)

class Sparse(nn.Linear):
    def sparse_forward(self, x, iter_type, iter_num, eps, types='relu', act_args=0.1):
        return getattr(self, iter_type + '_forward')(x, iter_num, eps, types, act_args)

    def true_forward(self, x, iter_type, iter_num, eps, types='relu', act_args=0.1):
        a = getattr(self, iter_type + '_forward')(x, iter_num, eps, types, act_args)
        index = (a > 0).detach()
        a_true = a.new_zeros()
        return None

    def down_forward(self, x, iter_num, eps, types, act_args):
        G = Tensor.matmul(self.weight, self.weight.t())
        G = G - Tensor.diag(Tensor.diag(G))
        bt = F.linear(x, self.weight, None)
        ut = bt
        for i in range(iter_num):
            ut = (1 - eps) * ut + eps * (bt - Tensor.matmul(act(ut, types=types, act_args=act_args), G))
        return act(ut, types=types, act_args=act_args)

    def grow_forward(self, x, iter_num, eps, types, act_args):
        G = Tensor.matmul(self.weight, self.weight.t())
        G = G - Tensor.diag(Tensor.diag(G))
        bt = F.linear(x, self.weight, None)
        if iter_num == 0:
            return act(bt, types=types, act_args=act_args)
        ut = torch.zeros_like(bt)
        for i in range(iter_num):
            ut = (1 - eps) * ut + eps * (bt - Tensor.matmul(act(ut, types=types, act_args=act_args), G))
        return act(ut, types=types, act_args=act_args)

    def another_forward(self, x, iter_num, eps, types, act_args):
        G = Tensor.matmul(self.weight, self.weight.t())
        G = G - Tensor.diag(Tensor.diag(G))
        bt = F.linear(x, self.weight, None)
        ut = bt
        for i in range(iter_num):
            ut = act(ut + eps * (bt - Tensor.matmul(ut, G) - ut), types=types, act_args=act_args)
        return ut

class MSC(nn.Module):
    def __init__(self, layer_sizes, iter_type='down', iter_num=0, eps=0.5, act_args=0.1):
        super(MSC, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.iter_num = iter_num
        self.eps = eps
        self.nlayer = len(self.layer_sizes)-1
        self.fcs = nn.ModuleList([Sparse(self.layer_sizes[l], self.layer_sizes[l+1], bias=False) for l in range(self.nlayer)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(self.layer_sizes[l+1], momentum=0.1, affine=False) for l in range(self.nlayer)])
        for fc in self.fcs:
            fc.weight.data /= (fc.weight**2).sum(1,keepdim=True).sqrt()

    def normal_forward(self, x):
        x = x.view(-1, 784)
        for i in range(self.nlayer):
            x = self.fcs[i](x)
            if i < self.nlayer-1:
                x = self.bn[i](x)
                x = F.relu(x)
        return x

    def forward_layers(self, x, rt_layers = False):
        layers = []
        x = x.view(-1, 784)
        layers.append(x)
        for i in range(self.nlayer):
            if i < self.nlayer-1:
                x = self.fcs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps, types=self.types, act_args=self.act_args)
                # x = self.bn[i](x)
                # x = act(x, types=self.types, act_args=self.act_args)
            else:
                x = self.fcs[i](x)
            layers.append(x)
        if rt_layers:
            return layers
        else:
            return x

class SparseConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, gk=0.0):
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.n_dim = 2
        self.check = 3
        self.padding_G = [int(np.floor((k - 1) / s) * s) for k, s in zip(self.kernel_size, self.stride)]
        self.padding_r = [k - 1 - p for k, p in zip(self.kernel_size, self.padding)]
        self.gk = gk
#         Computing the normalization coefficient
        self.bsq = np.zeros(self.kernel_size)
        self.Rho = []
        for k_w in range(1, self.kernel_size[0] + 1):
            for k_h in range(1, self.kernel_size[1] + 1):
                k_s = k_w * k_h
                pos = np.stack([np.tile(np.arange(k_w), [k_h, 1]), np.tile(np.arange(k_h), [k_w, 1]).transpose()], axis=2)
                pos = pos.reshape([k_s, 1, 2]).repeat(k_s, 1)
                pos = np.concatenate([pos, pos.transpose([1, 0, 2])], axis=2)
                dis = np.abs(pos[:, :, 1] - pos[:, :, 3]) + np.abs(pos[:, :, 0] - pos[:, :, 2])
                Rho_i = np.exp(-self.gk * dis)
                self.Rho.append(Rho_i)
                if gk == 0:
                    self.bsq[k_w - 1, k_h - 1] = np.float(1)
                else:
                    self.bsq[k_w - 1, k_h - 1] = np.mat(Rho_i).I.sum()
                with torch.no_grad():
                    self.gauss_kernel = torch.Tensor(range(-self.padding_G[0], self.padding_G[0] + 1)).abs()
                    self.gauss_kernel = self.gauss_kernel.reshape(1, -1) + self.gauss_kernel.reshape(-1, 1)
                    self.gauss_kernel = (self.gauss_kernel*(-gk)).exp()

    def sparse_forward(self, x, iter_type, iter_num, eps, types='relu', act_args=0.1):
        return getattr(self, iter_type + '_forward')(x, iter_num, eps, types, act_args)

    # def true_forward(self, x, iter_type, iter_num, eps, types='relu', act_args=0.1):
    #     a = getattr(self, iter_type + '_forward')(x, iter_num, eps, types, act_args)
    #     index = (a > 0).detach()
    #     a_true = a.new_zeros()

    #     return None

    def down_forward(self, x, iter_num, eps, types, act_args):
        G = F.conv2d(self.weight, self.weight, bias=None, stride=self.stride, padding=self.padding_G, dilation=1, groups=1)
        G = G * self.gauss_kernel.to(G)
        bt = self.forward(x)
        ut = bt
        for i in range(iter_num):
            at = act(ut, types=types, act_args=act_args)
            ut = (1 - eps) * ut + eps * (bt + at - F.conv2d(at, G, bias=None, stride=1, padding=self.padding_G, dilation=self.stride, groups=1))
        return act(ut, types=types, act_args=act_args)


    def recons(self, x, mask=True, bias=True):
        # w_t = self.weight.permute(1,0,2,3).flip(2,3)
        im_s = x.shape[2:4]
        im_s = [im_s[i] + 2 * self.padding_r[i] + 1 - self.kernel_size[i] for i in range(len(im_s))]

        try:
            if self.recons_mask.shape != torch.Size(im_s):
                raise AttributeError
        except AttributeError:
            sp = x.shape[2:4]
            fin = []
            for a, p, k in zip(sp, self.padding_r, self.kernel_size):
                ind = np.arange(a + 2 * p - k + 1)
                ind = np.minimum(np.minimum(ind + k - p, k), np.minimum(a + p - ind, k))
                fin.append(ind - 1)
            recons_mask = x.new(size=im_s)
            for i in range(recons_mask.shape[0]):
                for j in range(recons_mask.shape[1]):
                    recons_mask[i, j] = 1 / self.bsq[fin[0][i], fin[1][j]]
            self.recons_mask = recons_mask

        if bias:
            # out_raw = F.conv2d(x, w_t, bias=None, stride=1, padding=self.padding_r, dilation=1, groups=1)
            out_raw = F.conv_transpose2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)
            if not mask:
                return out_raw
            else:
                return self.recons_mask * out_raw
        else:
            w_t = self.weight.permute(1,0,2,3).flip(2,3)
            b_s = x.size(0)
            k_s = self.weight.shape[2:4].numel()
            x_unf = F.unfold(x, self.kernel_size, padding=self.padding_r)
            x_unf = x_unf.view(b_s, self.out_channels, k_s, -1).permute(0, 2, 3, 1).matmul(w_t.reshape(self.in_channels, self.out_channels, -1).permute(2, 1, 0)).permute(0, 2, 3, 1)
            # size is [N, H*W, C_in, k_s]
            x_unf = F.fold(x_unf.matmul(x_unf.new(self.Rho[-1])).mul(x_unf).sum(3).transpose(1, 2), self.recons_mask.shape, 1)
            # x_unf = (x_unf * self.recons_mask).sqrt() * out_raw.sign()
            # x_unf = x_unf / out_raw
            x_unf = (x_unf * self.recons_mask).sqrt()
            return x_unf

    def energy_c(self, x, reduce=True):
        w = self.weight.permute(1, 0, 2, 3).flip(2, 3)
        k_i = w.shape[0]
        k_o = w.shape[1]
        k_h = w.shape[2]
        k_w = w.shape[3]
        k_s = k_w * k_h
        w = w.reshape(k_i, k_o, k_s)
        w_ex = w.new_zeros(k_i, k_o, k_s, k_s)
        for i in range(k_s):
            w_ex[:, :, i, i] = w[:, :, i]
        w_ex = w_ex.permute(0, 3, 1, 2).resize(k_i*k_s, k_o, k_h, k_w)
        rec_d = F.conv2d(x, w_ex, bias=None, stride=1, padding=self.padding_r, dilation=1, groups=1)
        rec_d = rec_d.reshape(rec_d.shape[0], k_i, k_s, rec_d.shape[2], rec_d.shape[3]).permute(0,1,3,4,2)
        Rho = torch.from_numpy(self.Rho[-1]).to(rec_d)
        Rho = Rho - Rho.new_ones(Rho.shape) / self.bsq[-1, -1]
        if reduce:
            return (rec_d * rec_d.matmul(Rho)).sum()
        else:
            return (rec_d * rec_d.matmul(Rho))


class CSC_CIFAR10(nn.Module):
    def __init__(self, layer_sizes=(3, 64, 64, 128, 128), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0):
        super(CSC_CIFAR10, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [3, 32, 32]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([SparseConv(self.layer_sizes[l], self.layer_sizes[l + 1], kernel_size=3, padding=1, bias=False, gk=self.gk) for l in range(self.nlayers)])
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(self.nlayers + 1)])
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()
        self.fc1 = nn.Linear(self.layer_sizes[-1] * 8 * 8, 200)
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, *self.input_size)
        layers.append(x)
        for i in range(self.nlayers):
            # if i == 3:
            #     x = self.convs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=0, eps=self.eps,
            #                                types=self.types, act_args=0)
            # else:
            #     x = self.convs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps,
            #                                      types=self.types, act_args=self.act_args)
            x = self.scbns[i](x)
            x = self.scconvs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps,
                                               types=self.types, act_args=self.act_args)
            layers.append(x)
            if i % 2 == 1:
                x, index = F.max_pool2d(x, 2, return_indices=True)
                indices.append(index)
                layers.append(x)

        x = self.scbns[-1](x)
        x = x.view(-1, self.layer_sizes[-1] * 8 * 8)
        x = F.relu(self.fc1(x))
        layers.append(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class CSC_MNIST(nn.Module):
    def __init__(self, layer_sizes=(1, 32, 32, 64, 64), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0):
        super(CSC_MNIST, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [1, 28, 28]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([SparseConv(self.layer_sizes[l], self.layer_sizes[l + 1], kernel_size=3, padding=1, bias=False, gk=self.gk) for l in range(self.nlayers)])
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(self.nlayers + 1)])
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()
        self.fc1 = nn.Linear(self.layer_sizes[-1] * 7 * 7, 200)
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, *self.input_size)
        layers.append(x)
        for i in range(self.nlayers):
            x = self.scbns[i](x)
            x = self.scconvs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps,
                                               types=self.types, act_args=self.act_args)
            layers.append(x)
            if i % 2 == 1:
                x, index = F.max_pool2d(x, 2, return_indices=True)
                indices.append(index)
                layers.append(x)

        x = self.scbns[-1](x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        layers.append(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x

class CSC_try(nn.Module):
    def __init__(self, layer_sizes=(3, 64, 64, 128, 128), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0, sc_r=1):
        super(CSC_try, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [3, 32, 32]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.sc_r = sc_r
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([SparseConv(self.layer_sizes[l], self.layer_sizes[l]*self.sc_r, kernel_size=3, padding=1, bias=False, gk=self.gk) for l in range(1, self.nlayers+1, 1)])
        self.fconvs = nn.ModuleList([nn.Conv2d(self.layer_sizes[l], self.layer_sizes[l+1], kernel_size=3, padding=1, bias=False) for l in range(self.nlayers)])
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(self.nlayers+1)])
        self.fbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l+1], affine=False) for l in range(self.nlayers)])
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()
        self.fc1 = nn.Linear(self.layer_sizes[-1] * 8 * 8, 200)
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)


    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        layers.append(x)
        x = x.view(-1, *self.input_size)
        for i in range(self.nlayers):
            x = self.fconvs[i](x)
            layers.append(x)
            x = self.scconvs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps,
                                             types=self.types, act_args=self.act_args)
            layers.append(x)
            x = self.scconvs[i].recons(x)
            x = self.fbns[i](x)
            x = F.relu(x)
            if i % 2 == 1:
                x, index = F.max_pool2d(x, 2, return_indices=True)
                indices.append(index)
                layers.append(x)
        x = x.view(-1, self.layer_sizes[-1] * 8 * 8)
        x = F.relu(self.fc1(x))
        layers.append(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class CSC_try1(nn.Module):
    def __init__(self, layer_sizes=(3, 64, 64, 128, 128), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0):
        super(CSC_try1, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [3, 32, 32]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([SparseConv(self.layer_sizes[l], self.layer_sizes[l + 1], kernel_size=3, padding=1, bias=False, gk=self.gk) for l in range(self.nlayers)])
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(self.nlayers + 1)])
        self.fbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(1, self.nlayers + 1)])
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()
        self.fc1 = nn.Linear(self.layer_sizes[-1] * 8 * 8, 200)
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)


    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, *self.input_size)
        layers.append(x)
        for i in range(self.nlayers):
            if i < 3:
                x = self.scconvs[i](x)
                x = self.fbns[i](x)
                x = F.relu(x)
                layers.append(x)
            else:

                # x = self.scbns[i](x)
                x = self.scconvs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps,
                                                   types=self.types, act_args=self.act_args)

                layers.append(x)
            if i % 2 == 1:
                x, index = F.max_pool2d(x, 2, return_indices=True)
                indices.append(index)
                layers.append(x)

        x = x.view(-1, self.layer_sizes[-1] * 8 * 8)
        x = F.relu(self.fc1(x))
        layers.append(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class CNN_plain(nn.Module):
    def __init__(self, layer_sizes=(3, 64, 64, 128, 128), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0, sc_r=1):
        super(CNN_plain, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [3, 32, 32]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.sc_r = sc_r
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([nn.Conv2d(self.layer_sizes[l], self.layer_sizes[l]*self.sc_r, kernel_size=3, padding=1, bias=False) for l in range(1, self.nlayers+1, 1)])
        self.fconvs = nn.ModuleList([nn.Conv2d(self.layer_sizes[0], self.layer_sizes[1], kernel_size=3, padding=1, bias=False)] + [nn.Conv2d(self.layer_sizes[l]*self.sc_r, self.layer_sizes[l+1], kernel_size=3, padding=1, bias=False) for l in range(1, self.nlayers)])
        self.last_conv = nn.Conv2d(self.layer_sizes[-1]*self.sc_r, self.layer_sizes[-1], kernel_size=3, padding=1, bias=False)
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l+1]*self.sc_r, affine=False) for l in range(self.nlayers)])
        self.fbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l+1], affine=False) for l in range(self.nlayers)])
        self.last_bn = nn.BatchNorm2d(self.layer_sizes[-1], affine=False)
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()
        self.fc1 = nn.Linear(self.layer_sizes[-1] * 8 * 8, 200)
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        layers.append(x)
        x = x.view(-1, *self.input_size)
        for i in range(self.nlayers):
            x = self.fconvs[i](x)
            x = self.fbns[i](x)
            x = F.relu(x)
            layers.append(x)
            x = self.scconvs[i](x)
            x = self.scbns[i](x)
            x = F.relu(x)
            layers.append(x)
            if i == 3:
                x = self.last_conv(x)
                x = self.last_bn(x)
                x = F.relu(x)
                layers.append(x)
            if i % 2 == 1:
                x, index = F.max_pool2d(x, 2, return_indices=True)
                indices.append(index)
        x = x.view(-1, self.layer_sizes[-1] * 8 * 8)
        x = F.relu(self.fc1(x))
        layers.append(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class CSC_try2(nn.Module):
    def __init__(self, layer_sizes=(3, 64, 64, 128, 128), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0):
        super(CSC_try2, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [3, 32, 32]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([SparseConv(self.layer_sizes[l], self.layer_sizes[l + 1], kernel_size=3, padding=1, bias=False, gk=self.gk) for l in range(self.nlayers)])
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(self.nlayers + 1)])
        self.fbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(1, self.nlayers + 1)])
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()
        self.fc1 = nn.Linear(self.layer_sizes[-1] * 8 * 8, 200)
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)


    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, *self.input_size)
        layers.append(x)
        for i in range(self.nlayers):
            if i % 2 == 0:
                x = self.scconvs[i](x)
                x = self.fbns[i](x)
                x = F.relu(x)
                layers.append(x)
            else:

                # x = self.scbns[i](x)
                x = self.scconvs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps,
                                                   types=self.types, act_args=self.act_args)

                layers.append(x)
                x, index = F.max_pool2d(x, 2, return_indices=True)
                indices.append(index)
                layers.append(x)

        x = x.view(-1, self.layer_sizes[-1] * 8 * 8)
        x = F.relu(self.fc1(x))
        layers.append(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class CSC_try3(nn.Module):
    def __init__(self, layer_sizes=(3, 64, 64, 128, 128), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0):
        super(CSC_try3, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [3, 32, 32]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([SparseConv(self.layer_sizes[l], self.layer_sizes[l + 1], kernel_size=3, padding=1, bias=False, gk=self.gk) for l in range(self.nlayers)])
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(self.nlayers + 1)])
        self.fbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=True) for l in range(1, self.nlayers + 1)])
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()
        self.fc1 = nn.Linear(self.layer_sizes[-1] * 8 * 8, 200)
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, *self.input_size)
        layers.append(x)
        for i in range(self.nlayers):
            if i % 2 == 0:
                x = self.scconvs[i](x)
                x = self.fbns[i](x)
                x = F.relu(x)
                layers.append(x)
            else:

                x = self.scbns[i](x)
                x = self.scconvs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps,
                                                   types=self.types, act_args=self.act_args)

                layers.append(x)
                x, index = F.max_pool2d(x, 2, return_indices=True)
                indices.append(index)
                layers.append(x)

        x = x.view(-1, self.layer_sizes[-1] * 8 * 8)
        x = F.relu(self.fc1(x))
        layers.append(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class CSC_plain3(nn.Module):
    def __init__(self, layer_sizes=(3, 64, 64, 128, 128), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0):
        super(CSC_plain3, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [3, 32, 32]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([SparseConv(self.layer_sizes[l], self.layer_sizes[l + 1], kernel_size=3, padding=1, bias=False, gk=self.gk) for l in range(self.nlayers)])
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(self.nlayers + 1)])
        self.fbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(1, self.nlayers + 1)])
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()
        self.fc1 = nn.Linear(self.layer_sizes[-1] * 8 * 8, 200)
        # self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x, rt_layers=False):
        layers = []
        indices = []
        x = x.view(-1, *self.input_size)
        layers.append(x)
        for i in range(self.nlayers):
            if i % 2 == 0:
                x = self.scconvs[i](x)
                x = self.fbns[i](x)
                x = F.relu(x)
                layers.append(x)
            else:
                x = self.scconvs[i](x)
                x = self.fbns[i](x)
                x = F.relu(x)
                layers.append(x)
                x, index = F.max_pool2d(x, 2, return_indices=True)
                indices.append(index)
                layers.append(x)

        x = self.scbns[-1](x)
        x = x.view(-1, self.layer_sizes[-1] * 8 * 8)
        x = F.relu(self.fc1(x))
        layers.append(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        layers.append(x)
        if rt_layers:
            return layers, indices
        else:
            return x


class SC_CNN(nn.Module):
    def __init__(self, layer_sizes=(1, 32), iter_type='down', iter_num=0, eps=0.5, act_args=0.1, gk=0.0, kernel_sizes=None, paddings=None):
        super(SC_CNN, self).__init__()
        self.iter_type = iter_type
        self.types = 'soft_single'
        self.act_args = act_args
        self.layer_sizes = layer_sizes
        self.input_size = [1, 28, 28]
        self.gk = gk
        self.iter_num = iter_num
        self.eps = eps
        self.kernel_sizes = kernel_sizes
        self.nlayers = len(self.layer_sizes) - 1
        self.scconvs = nn.ModuleList([SparseConv(self.layer_sizes[l], self.layer_sizes[l + 1], kernel_size=kernel_sizes[l], padding=paddings[l], bias=False, gk=self.gk) for l in range(self.nlayers)])
        self.scbns = nn.ModuleList([nn.BatchNorm2d(self.layer_sizes[l], affine=False) for l in range(self.nlayers + 1)])
        for conv in self.scconvs:
            conv.weight.data /= (conv.weight**2).sum([1, 2, 3], keepdim=True).sqrt()


    def forward(self, x, rt_layers=False):
        x = x.view(-1, *self.input_size)
        layers = [x, ]
        for i in range(self.nlayers):
            x = self.scconvs[i].sparse_forward(x, iter_type=self.iter_type, iter_num=self.iter_num, eps=self.eps,
                                               types=self.types, act_args=self.act_args)
            layers.append(x)
        if rt_layers:
            return layers
        else:
            return x


    def recons(self, x):
        for i in range(self.nlayers - 1, -1, -1):
            x = self.scconvs[i].recons(x)
        return x
