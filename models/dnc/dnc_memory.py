#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os, sys

from models.meta.memory import Memory
from models.utils import *


'''
Class of the DNC External Memory Module.
'''
class DNCMemory(Memory):
    def __init__(self, args):
        super(DNCMemory, self).__init__(args)
        self.write_heads = 1
        self.independent_linears = args.independent_linears
        m = self.num_address
        w = self.mem_cell_dim
        r = self.read_heads
        self.interface_size = (w * r) + (3 * w) + (5 * r) + 3

        if self.independent_linears:
            self.read_keys_transform = nn.Linear(self.interface_size, w * r)
            self.read_strengths_transform = nn.Linear(self.interface_size, r)
            self.write_key_transform = nn.Linear(self.interface_size, w)
            self.write_strength_transform = nn.Linear(self.interface_size, 1)
            self.erase_vector_transform = nn.Linear(self.interface_size, w)
            self.write_vector_transform = nn.Linear(self.interface_size, w)
            self.free_gates_transform = nn.Linear(self.interface_size, r)
            self.allocation_gate_transform = nn.Linear(self.interface_size, 1)
            self.write_gate_transform = nn.Linear(self.interface_size, 1)
            self.read_modes_transform = nn.Linear(self.interface_size, 3 * r)
        
        self.I = t_cuda(torch.ones(1,m,m) - torch.eye(m).unsqueeze(0), self.use_cuda)
    
    def reset(self, hidden=None, erase=True):
        m = self.num_address
        w = self.mem_cell_dim
        r = self.read_heads
        b = self.batch_size
        
        if hidden is None:
            _output = {
                'memory': t_cuda(torch.zeros(b, m, w).fill_(0), self.use_cuda),
                'link_matrix': t_cuda(torch.zeros(b, 1, m, m), self.use_cuda),
                'precedence': t_cuda(torch.zeros(b, 1, m), self.use_cuda),
                'read_weights': t_cuda(torch.zeros(b, r, m).fill_(0), self.use_cuda),
                'write_weights': t_cuda(torch.zeros(b, 1, m).fill_(0), self.use_cuda),
                'usage_vector': t_cuda(torch.zeros(b, m), self.use_cuda),
                'read_modes': None,
                'allocation_gate': None,
                'free_gates': None,
                'read_vectors': None
            }
            if self.memory_init_style == "constant":
                _output['memory'].data.fill_(0.000001)
            elif self.memory_init_style == "random":
                _output['memory'] = t_cuda(torch.zeros(b, m, w).normal_(std=1./float(m*w)), self.use_cuda)
            else:
                pass
            return _output
        else:
            hidden['memory'] = hidden['memory'].clone()
            hidden['link_matrix'] = hidden['link_matrix'].clone()
            hidden['precedence'] = hidden['precedence'].clone()
            hidden['read_weights'] = hidden['read_weights'].clone()
            hidden['write_weights'] = hidden['write_weights'].clone()
            hidden['usage_vector'] = hidden['usage_vector'].clone()
            hidden['read_modes'] = hidden['read_modes'].clone()
            hidden['allocation_gate'] = hidden['allocation_gate'].clone()
            hidden['free_gates'] = hidden['free_gates'].clone()
            hidden['read_vectors'] = hidden['read_vectors'].clone()

            if erase:
                hidden['memory'].data.fill_(0)
                hidden['link_matrix'].data.zero_()
                hidden['precedence'].data.zero_()
                hidden['read_weights'].data.fill_(0)
                hidden['write_weights'].data.fill_(0)
                hidden['usage_vector'].data.zero_()
            
            return hidden
    
    ####################### sub functions for write() #######################
    def get_usage_vector(self, usage, free_gates, read_weights, write_weights):
        # write_weights = write_weights.detach()  # detach from the computation graph
        usage = usage + (1 - usage) * (1 - torch.prod(1 - write_weights, 1))
        ψ = torch.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
        return usage * ψ
    
    def allocate(self, usage, write_gate):
        # ensure values are not too small prior to cumprod.
        usage = δ + (1 - δ) * usage
        batch_size = usage.size(0)
        # free list
        sorted_usage, φ = torch.topk(usage, self.num_address, dim=1, largest=False)
        
        # cumprod with exclusive=True
        # https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
        v = sorted_usage.data.new(batch_size, 1).fill_(1).requires_grad_()
        cat_sorted_usage = torch.cat((v, sorted_usage), 1)
        prod_sorted_usage = torch.cumprod(cat_sorted_usage, 1)[:, :-1]
        
        sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()
        # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
        _, φ_rev = torch.topk(φ, k=self.num_address, dim=1, largest=False)
        allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())
        
        return allocation_weights.unsqueeze(1), usage
    
    def write_weighting(self, memory, write_content_weights, allocation_weights, write_gate, allocation_gate):
        ag = allocation_gate.unsqueeze(-1)
        wg = write_gate.unsqueeze(-1)
        return wg * (ag * allocation_weights + (1 - ag) * write_content_weights)
    
    def get_link_matrix(self, link_matrix, write_weights, precedence):
        precedence = precedence.unsqueeze(2)
        write_weights_i = write_weights.unsqueeze(3)
        write_weights_j = write_weights.unsqueeze(2)
        
        prev_scale = 1 - write_weights_i - write_weights_j
        new_link_matrix = write_weights_i * precedence
        
        link_matrix = prev_scale * link_matrix + new_link_matrix
        # trick to delete diag elems
        return self.I.expand_as(link_matrix) * link_matrix
    
    def update_precedence(self, precedence, write_weights):
        return (1 - torch.sum(write_weights, 2, keepdim=True)) * precedence + write_weights
    #########################################################################

    ####################### write() #######################
    def write(
        self,
        write_key,
        write_vector,
        erase_vector,
        free_gates,
        read_strengths,
        write_strength,
        write_gate,
        allocation_gate,
        hidden
    ):
        # get current usage
        hidden['usage_vector'] = self.get_usage_vector(
            hidden['usage_vector'],
            free_gates,
            hidden['read_weights'],
            hidden['write_weights']
        )
        hidden['allocation_gate'] = allocation_gate
        hidden['free_gates'] = free_gates
        # lookup memory with write_key and write_strength
        write_content_weights = self.content_weightings(hidden['memory'], write_key, write_strength)
        
        # get memory allocation
        alloc, _ = self.allocate(
            hidden['usage_vector'],
            allocation_gate * write_gate
        )
        
        # get write weightings
        hidden['write_weights'] = self.write_weighting(
            hidden['memory'],
            write_content_weights,
            alloc,
            write_gate,
            allocation_gate
        )
        
        weighted_resets = hidden['write_weights'].unsqueeze(3) * erase_vector.unsqueeze(2)
        reset_gate = torch.prod(1 - weighted_resets, 1)
        # Update memory
        hidden['memory'] = hidden['memory'] * reset_gate
        
        hidden['memory'] = hidden['memory'] + torch.bmm(hidden['write_weights'].transpose(1, 2), write_vector)
        
        # update link_matrix
        hidden['link_matrix'] = self.get_link_matrix(
            hidden['link_matrix'],
            hidden['write_weights'],
            hidden['precedence']
        )
        
        hidden['precedence'] = self.update_precedence(hidden['precedence'], hidden['write_weights'])
        return hidden
    #######################################################
    
    ####################### sub functions for read() #######################
    def content_weightings(self, memory, keys, strengths):
        d = θ(memory, keys)
        return σ(d * strengths.unsqueeze(2), 2)
    
    def directional_weightings(self, link_matrix, read_weights):
        rw = read_weights.unsqueeze(1)
        
        f = torch.matmul(link_matrix, rw.transpose(2, 3)).transpose(2, 3)
        b = torch.matmul(rw, link_matrix)
        return f.transpose(1, 2), b.transpose(1, 2)
    
    def read_weightings(self, memory, content_weights, link_matrix, read_modes, read_weights):
        forward_weight, backward_weight = self.directional_weightings(link_matrix, read_weights)
        
        content_mode = read_modes[:, :, 2].contiguous().unsqueeze(2) * content_weights
        backward_mode = torch.sum(read_modes[:, :, 0:1].contiguous().unsqueeze(3) * backward_weight, 2)
        forward_mode = torch.sum(read_modes[:, :, 1:2].contiguous().unsqueeze(3) * forward_weight, 2)
        
        return backward_mode + content_mode + forward_mode
    
    def read_vectors(self, memory, read_weights):
        return torch.bmm(read_weights, memory)
    ########################################################################

    ####################### read() #######################
    def read(self, read_keys, read_strengths, read_modes, hidden):
        content_weights = self.content_weightings(hidden['memory'], read_keys, read_strengths)
        hidden["read_modes"] = read_modes
        hidden['read_weights'] = self.read_weightings(
            hidden['memory'],
            content_weights,
            hidden['link_matrix'],
            read_modes,
            hidden['read_weights']
        )
        read_vectors = self.read_vectors(hidden['memory'], hidden['read_weights'])
        hidden["read_vectors"] = read_vectors
        return read_vectors, hidden
    ######################################################
    
    ####################### forward() #######################
    def forward(self, ξ, hidden):
        m = self.num_address
        w = self.mem_cell_dim
        r = self.read_heads
        b = self.batch_size
        
        ## Create parameters
        if self.independent_linears:
            read_keys = F.tanh(self.read_keys_transform(ξ).view(b, r, w)) # r read keys (b * r * w)
            read_strengths = F.softplus(self.read_strengths_transform(ξ).view(b, r)) # r read strengths (b * r)
            write_key = F.tanh(self.write_key_transform(ξ).view(b, 1, w)) # write key (b * 1 * w)
            write_strength = F.softplus(self.write_strength_transform(ξ).view(b, 1)) # write strength (b * 1)
            erase_vector = F.sigmoid(self.erase_vector_transform(ξ).view(b, 1, w)) # erase vector (b * 1 * w)
            write_vector = F.tanh(self.write_vector_transform(ξ).view(b, 1, w)) # write vector (b * 1 * w)
            free_gates = F.sigmoid(self.free_gates_transform(ξ).view(b, r)) # r free gates (b * r)
            allocation_gate = F.sigmoid(self.allocation_gate_transform(ξ).view(b, 1)) # allocation gate (b * 1)
            write_gate = F.sigmoid(self.write_gate_transform(ξ).view(b, 1)) # write gate (b * 1)
            read_modes = σ(self.read_modes_transform(ξ).view(b, r, 3), 1) # read modes (b * r * 3)
        else:
            read_keys = F.tanh(ξ[:, :r * w].contiguous().view(b, r, w)) # r read keys (b * w * r)
            read_strengths = F.softplus(ξ[:, r * w:r * w + r].contiguous().view(b, r)) # r read strengths (b * r)
            write_key = F.tanh(ξ[:, r * w + r:r * w + r + w].contiguous().view(b, 1, w)) # write key (b * w * 1)
            write_strength = F.softplus(ξ[:, r * w + r + w].contiguous().view(b, 1)) # write strength (b * 1)
            erase_vector = F.sigmoid(ξ[:, r * w + r + w + 1: r * w + r + 2 * w + 1].contiguous().view(b, 1, w)) # erase vector (b * w)
            write_vector = F.tanh(ξ[:, r * w + r + 2 * w + 1: r * w + r + 3 * w + 1].contiguous().view(b, 1, w)) # write vector (b * w)
            free_gates = F.sigmoid(ξ[:, r * w + r + 3 * w + 1: r * w + 2 * r + 3 * w + 1].contiguous().view(b, r)) # r free gates (b * r)
            allocation_gate = F.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 1].contiguous().unsqueeze(1).view(b, 1)) # allocation gate (b * 1)
            write_gate = F.sigmoid(ξ[:, r * w + 2 * r + 3 * w + 2].contiguous()).unsqueeze(1).view(b, 1) # write gate (b * 1)
            read_modes = σ(ξ[:, r * w + 2 * r + 3 * w + 3: r * w + 5 * r + 3 * w + 3].contiguous().view(b, r, 3), 1) # read modes (b * 3*r)
        
        ## Write
        hidden = self.write(
            write_key,
            write_vector,
            erase_vector,
            free_gates,
            read_strengths,
            write_strength,
            write_gate,
            allocation_gate,
            hidden
        )
        
        ## Read
        return self.read(read_keys, read_strengths, read_modes, hidden)
    #########################################################