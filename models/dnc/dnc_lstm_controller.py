#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os, sys

from models.meta.controller import Controller
from models.utils import *

class DNCLstmController(Controller):
    def __init__(self, args):
        super(DNCLstmController, self).__init__(args)
        self.in_to_hid = nn.Linear(self.input_dim, self.hidden_dim)
        self.mem_to_hid = nn.Linear(self.read_vec_dim, self.hidden_dim)
        layers = []
        layers.append(nn.Linear(2*self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        self.linears = nn.Sequential(*layers)
        self.rnn = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            bias=True,
            batch_first=True,
            num_layers=self.num_hid_layers,
            dropout=self.dropout_value
        )
        m = self.num_address
        w = self.mem_cell_dim
        r = self.read_heads
        self.interface_size = (w * r) + (3 * w) + (5 * r) + 3
        self.hid_to_mem = nn.Linear(self.hidden_dim, self.interface_size)
        self.hid_to_out = nn.Linear(self.hidden_dim, self.output_dim)

    def _init_weights(self):
        pass
    
    def forward(self, x, read, hx):
        x = self.in_to_hid(x)
        read = self.mem_to_hid(read)
        input_vec = torch.cat((x, read), dim=-1)
        output_vec = self.linears(input_vec)
        output_vec, hx = self.rnn(output_vec, hx)
        out_to_mem_vec = self.hid_to_mem(output_vec)
        output_vec = self.hid_to_out(output_vec)
        return out_to_mem_vec, output_vec, hx