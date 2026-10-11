"""Small torch-native subset of einops used by this repository.

It covers only the fixed patterns present in the checked-in model files. This
keeps the training environment self-contained without changing tensor order.
"""

import torch
from torch import nn


def rearrange(x, pattern, **axes):
    pattern = ' '.join(pattern.split())
    if pattern == 'b n (h d) -> b h n d':
        b, n, hd = x.shape
        h = axes['h']
        return x.reshape(b, n, h, hd // h).permute(0, 2, 1, 3)
    if pattern == 'b h n d -> b n (h d)':
        b, h, n, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, n, h * d)
    if pattern == 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)':
        b, c, hp1, wp2 = x.shape
        p1, p2 = axes['p1'], axes['p2']
        h, w = hp1 // p1, wp2 // p2
        return x.reshape(b, c, h, p1, w, p2).permute(0, 2, 4, 3, 5, 1).reshape(
            b, h * w, p1 * p2 * c)
    if pattern == 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)':
        b, hw, patch_dim = x.shape
        h, p1, p2 = axes['h'], axes['p1'], axes['p2']
        w = hw // h
        c = patch_dim // (p1 * p2)
        return x.reshape(b, h, w, p1, p2, c).permute(0, 5, 1, 3, 2, 4).reshape(
            b, c, h * p1, w * p2)
    if pattern == 'b c h w -> b (c w) h':
        b, c, h, w = x.shape
        return x.permute(0, 1, 3, 2).reshape(b, c * w, h)
    if pattern in ('b c h -> b h c', 'b w h -> b h w'):
        return x.transpose(1, 2)
    if pattern == 'c h w -> (c h w)':
        return x.reshape(-1)
    if pattern == 'b n (c h w) -> b n c h w':
        b, n, chw = x.shape
        c, h = axes['c'], axes['h']
        return x.reshape(b, n, c, h, chw // (c * h))
    raise NotImplementedError('Unsupported rearrange pattern: {}'.format(pattern))


def repeat(x, pattern, **axes):
    pattern = ' '.join(pattern.split())
    if pattern == '() n d -> b n d':
        return x.repeat(axes['b'], 1, 1)
    raise NotImplementedError('Unsupported repeat pattern: {}'.format(pattern))


class Rearrange(nn.Module):
    def __init__(self, pattern, **axes):
        super().__init__()
        self.pattern = pattern
        self.axes = axes

    def forward(self, x):
        return rearrange(x, self.pattern, **self.axes)
