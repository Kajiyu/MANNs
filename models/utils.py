#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os, sys

δ = 1e-6

def t_cuda(x, use_cuda):
    if use_cuda:
        return x.cuda()
    else:
        return x


def θ(a, b, dimA=2, dimB=2, normBy=2):
    """Batchwise Cosine distance

    Cosine distance

    Arguments:
        a {Tensor} -- A 3D Tensor (b * m * w)
        b {Tensor} -- A 3D Tensor (b * r * w)

    Keyword Arguments:
        dimA {number} -- exponent value of the norm for `a` (default: {2})
        dimB {number} -- exponent value of the norm for `b` (default: {1})

    Returns:
        Tensor -- Batchwise cosine distance (b * r * m)
    """
    a_norm = torch.norm(a, normBy, dimA, keepdim=True).expand_as(a) + δ
    b_norm = torch.norm(b, normBy, dimB, keepdim=True).expand_as(b) + δ

    x = torch.bmm(a, b.transpose(1, 2)).transpose(1, 2) / (
        torch.bmm(a_norm, b_norm.transpose(1, 2)).transpose(1, 2) + δ)
    # apply_dict(locals())
    return x


def σ(input, axis=1):
    """Softmax on an axis

    Softmax on an axis

    Arguments:
        input {Tensor} -- input Tensor

    Keyword Arguments:
        axis {number} -- axis on which to take softmax on (default: {1})

    Returns:
        Tensor -- Softmax output Tensor
    """
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d, -1)
    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)