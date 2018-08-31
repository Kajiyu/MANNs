#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os, sys


'''
Meta Class of the Controller Network.
'''
class Controller(nn.Module):
    def __init__(self, args):
        super(Controller, self).__init__()
        # logging
        self.logger = args.logger
        # general params
        self.use_cuda = args.use_cuda

        # params
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.read_vec_dim = args.read_vec_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.num_address = args.num_address
        self.mem_vec_dim = args.mem_vec_dim
        self.read_heads = args.read_heads
        self.write_heads = args.write_heads
        self.clip_value = args.clip_value
    
    def _init_weights(self):
        raise NotImplementedError("You must define `_init_weights` function if 'Controller' is based.")
    
    def forward(self, input_vb):
        raise NotImplementedError("You must define `forward` function if 'Controller' is based.")
    
    def _reset(self):           # NOTE: should be called at each child's __init__
        raise NotImplementedError("You must define `_reset` function if 'Controller' is based.")