import torch
from torch import nn
from collections import OrderedDict
import numpy as np

class Conv(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        # self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.conv(x)
        # out = self.bn(out)
        out = self.relu(out)
        return out

class Trojan(nn.Module):
    def __init__(self, h=None, w=None, sizes=(3, 32, 64, 64), repeat=2, down_sample=2):
        super(Trojan, self).__init__()
        self.h = [h]
        self.w = [w]
        self.sizes = sizes
        self.repeat = repeat
        self.down_sample = down_sample
        self.nlayer = len(self.sizes) - 1
        fwd = []
        for i in range(self.nlayer):
            group = []
            if i == 0:
                group.append(Conv(self.sizes[i], self.sizes[i+1], 1))
                self.h.append(h)
                self.w.append(w)
            else:
                group.append(Conv(self.sizes[i], self.sizes[i + 1], self.down_sample))
                self.h.append((self.h[-1]+1) // down_sample)
                self.w.append((self.w[-1]+1) // down_sample)

            for j in range(repeat - 1):
                group.append(Conv(self.sizes[i+1], self.sizes[i+1], 1))

            fwd.append(nn.Sequential(*group))

        self.fwd = nn.ModuleList(fwd)

        upsample = []
        bkw = []
        for i in range(self.nlayer - 1, 0, -1):
            group = []
            upsample.insert(0, nn.Upsample(size=(self.h[i], self.w[i]), mode='bilinear'))

            group.append(Conv(self.sizes[i+1] + self.sizes[i], self.sizes[i], 1))

            for j in range(repeat - 1):
                group.append(Conv(self.sizes[i], self.sizes[i], 1))

            bkw.insert(0, nn.Sequential(*group))

        self.upsample = nn.ModuleList(upsample)
        self.bkw = nn.ModuleList(bkw)
        self.final = nn.Conv2d(self.sizes[1], self.sizes[0], kernel_size=1, bias=False)

    def forward(self, x, rt_layers=False, pmin=-1, pmax=1):
        # x = x*2-1
        out = x
        outputs = []
        for i in range(self.nlayer):
            out = self.fwd[i](out)
            outputs.append(out)

        for i in range(self.nlayer - 2, -1, -1):
            out = self.upsample[i](out)
            # print(i)
            # print(out.shape)
            # print(outputs[i].shape)
            # print([o.shape for o in outputs])
            # print([w for w in self.w])
            # raise ValueError
            out = torch.cat((out, outputs[i]), 1)
            out = self.bkw[i](out)
        out = self.final(out)
        out += x
        # out = (out+1)/2
        # out = out.clamp(pmin, pmax)
        if rt_layers:
            return out, outputs
        else:
            return out



