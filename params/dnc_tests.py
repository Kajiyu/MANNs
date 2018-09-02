#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import visdom

import numpy as np
import os, sys

from params.meta import Params

class DNCTestParams(Params):
    def __init__(self):
        super(DNCTestParams, self).__init__()
        self.batch_size = 100
        self.num_address = 100
        self.mem_cell_dim = 10
        self.read_heads = 4
        self.write_heads = 1
        self.input_dim = 20 + 2
        self.output_dim = 20
        self.controller_input_dim = self.input_dim
        self.controller_output_dim = self.output_dim
        self.hidden_dim = 100
        self.num_hid_layers = 3
        self.dnc_output_activation = "tanh"
        self.memory_init_style = "constant"
        self.clip_value = 20
        self.controller_type = "ff_split"
        self.submodule = None
        self.dropout_value = 0.2
        self.independent_linears = True