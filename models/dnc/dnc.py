#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os, sys

from models.dnc.dnc_ff_controller import DNCFFController
from models.dnc.dnc_ff_split_controller import DNCFFSplitController
from models.dnc.dnc_rnn_controller import DNCRnnController
from models.dnc.dnc_lstm_controller import DNCLstmController
from models.dnc.dnc_gru_controller import DNCGruController
from models.dnc.dnc_memory import DNCMemory
from models.utils import *

class DNC(nn.Module):
    def __init__(self, args):
        super(DNC, self).__init__()
        # general params
        self.use_cuda = args.use_cuda
        # params
        self.batch_size = args.batch_size
        self.num_address = args.num_address
        self.mem_cell_dim = args.mem_cell_dim
        self.read_heads = args.read_heads
        self.write_heads = args.write_heads
        self.read_vec_dim = self.mem_cell_dim * self.read_heads
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.controller_input_dim = args.controller_input_dim
        self.controller_output_dim = args.controller_output_dim
        self.controller_hidden_dim = args.hidden_dim
        self.num_hid_layers = args.num_hid_layers
        self.dnc_output_activation = args.dnc_output_activation 

        m = self.num_address
        w = self.mem_cell_dim
        r = self.read_heads
        self.interface_size = (w * r) + (3 * w) + (5 * r) + 3

        self.clip_value = args.clip_value

        self.submodule = args.submodule
        self.controller_type = args.controller_type
        if self.controller_type == "ff":
            self.controller = DNCFFController(args)
        elif self.controller_type == "ff_split":
            self.controller = DNCFFSplitController(args)
        elif self.controller_type == "rnn":
            self.controller = DNCRnnController(args)
        elif self.controller_type == "lstm":
            self.controller = DNCLstmController(args)
        elif self.controller_type == "gru":
            self.controller = DNCGruController(args)
        self.memory = DNCMemory(args)
        self.integrate_output = nn.Linear(self.output_dim*2, self.output_dim)
        self.mem_output = nn.Linear(self.read_vec_dim, self.output_dim)
        self.con_output = nn.Linear(self.controller_output_dim, self.output_dim)

        self.mem_hidden = None
        if self.use_cuda:
            self.controller = self.controller.cuda()
    
    def init_hidden(self, hx, reset_experience):
        if hx is None:
            hx = (None, None, None)
        (chx, mhx, last_read) = hx
        if chx is None:
            if self.controller_type != "ff" and self.controller_type != "ff_split":
                h = t_cuda(torch.zeros(self.num_hid_layers, self.batch_size, self.controller_hidden_dim), self.use_cuda)
                xavier_uniform(h)
                chx = [ (h, h) if self.controller_type == 'lstm' else h for x in range(self.num_layers)]
        # Last read vectors
        if last_read is None:
            last_read = t_cuda(torch.zeros(self.batch_size, self.read_vec_dim), self.use_cuda)
        # memory states
        if mhx is None:
            mhx = self.memory.reset(hidden=None, erase=reset_experience)
        else:
            mhx = self.memory.reset(hidden=mhx, erase=reset_experience)
        return chx, mhx, last_read
    
    def debug(self, mhx, debug_obj):
        if not debug_obj:
            debug_obj = {
                'memory': [],
                'link_matrix': [],
                'precedence': [],
                'read_weights': [],
                'write_weights': [],
                'usage_vector': [],
                'read_modes': [],
                'allocation_gate': [],
                'free_gates': [],
                'read_vectors': [],
            }
        
        debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
        debug_obj['link_matrix'].append(mhx['link_matrix'][0][0].data.cpu().numpy())
        debug_obj['precedence'].append(mhx['precedence'][0].data.cpu().numpy())
        debug_obj['read_weights'].append(mhx['read_weights'][0].data.cpu().numpy())
        debug_obj['write_weights'].append(mhx['write_weights'][0].data.cpu().numpy())
        debug_obj['usage_vector'].append(mhx['usage_vector'][0].unsqueeze(0).data.cpu().numpy())
        debug_obj['read_modes'].append(mhx['read_modes'][0].data.cpu().numpy())
        debug_obj['allocation_gate'].append(mhx['allocation_gate'][0].unsqueeze(0).data.cpu().numpy())
        debug_obj['free_gates'].append(mhx['free_gates'][0].unsqueeze(0).data.cpu().numpy())
        debug_obj['read_vectors'].append(mhx['read_vectors'][0].data.cpu().numpy())
        return debug_obj

    def forward(self, input, hx=(None, None, None), output_activation=None):
        (chx, mhx, last_read) = hx
        if self.controller_type != "ff" and self.controller_type != "ff_split":
            out_to_mem_vec, output_vec, chx = self.controller(input, last_read, chx)
        else:
            out_to_mem_vec, output_vec = self.controller(input, last_read)
        if self.clip_value != 0:
            output_vec = torch.clamp(output_vec, -self.clip_value, self.clip_value)
        read_vecs, mhx = self.memory(out_to_mem_vec, mhx)
        read_vecs = read_vecs.view(-1, self.read_vec_dim)
        output_vec = self.con_output(output_vec)
        output_read_vecs = self.mem_output(read_vecs)
        output_vec = torch.cat((output_vec, output_read_vecs), dim=-1)
        output_vec = self.integrate_output(output_vec)
        if output_activation is None:
            output_activation = self.dnc_output_activation
        if output_activation == "relu":
            output_vec = nn.ReLU()(output_vec)
        elif output_activation == "tanh":
            output_vec = nn.Tanh()(output_vec)
        elif output_activation == "sigmoid":
            output_vec = nn.Sigmoid()(output_vec)
        elif output_activation == "softmax":
            output_vec = nn.Softmax()(output_vec)
        elif output_activation == "logsoftmax":
            output_vec = nn.LogSoftmax()(output_vec)
        else: # No Activation
            pass
        return output_vec, (chx, mhx, read_vecs)