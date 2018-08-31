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
Meta Class of the External Memory Module.
'''
class Memory(nn.Module):
    def __init__(self, args):
        super(Memory, self).__init__()
        # logging
        self.logger = args.logger
        # general params
        self.use_cuda = args.use_cuda

        # params
        self.batch_size = args.batch_size
        self.read_vec_dim = args.read_vec_dim
        self.num_address = args.num_address
        self.mem_vec_dim = args.mem_vec_dim
        self.read_heads = args.read_heads
        self.write_heads = args.write_heads
        self.memory_init_style = args.memory_init_style
    
    def write(self, w_vec):
        raise NotImplementedError("You must define `write` function if 'Memory' is based.")
    
    def read(self):
        raise NotImplementedError("You must define `read` function if 'Memory' is based.")
    
    def reset(self):
        raise NotImplementedError("You must define `reset` function if 'Controller' is based.")