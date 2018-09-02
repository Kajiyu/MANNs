#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os, sys

from models.dnc.dnc import DNC
from models.dnc.dnc_memory import DNCMemory
from models.utils import *

from params.dnc_tests import DNCTestParams

def onehot(x,n):
    ret = np.zeros(n).astype(np.float32)
    ret[x] = 1.0
    return ret

def generate_data(batch_size, length, size):
    input_data = []
    output_data = []
    for i in range(batch_size):
        seq_value = np.random.randint(size, size=length)
        order = np.random.randint(0, 2)
        sorted_content = sorted(seq_value, reverse=(order != 0))
        order = onehot(order, 2)
        input_seq = np.concatenate([order, np.zeros(size)]).reshape(1, -1)
        output_seq = np.zeros(((length+1, size)))
        for i in range(length):
            _input_vec = np.concatenate([np.zeros(2), onehot(seq_value[i], size)])
            _output_vec = onehot(sorted_content[i], size)
            input_seq = np.concatenate([input_seq, _input_vec.reshape(1, -1)])
            output_seq = np.concatenate([output_seq, _output_vec.reshape(1, -1)])
        input_seq = np.concatenate([input_seq, np.zeros((length, size+2))])
        input_data.append(input_seq.tolist())
        output_data.append(output_seq.tolist()) 
    input_data = np.array(input_data).astype("float32")
    target_output = np.array(output_data).astype("float32")
    return input_data, target_output

def criterion(predictions, targets):
    loss = (predictions - targets)**2
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)

'''
Tests for DNC
'''
def test_dnc():
    args = DNCTestParams()
    dnc = DNC(args)
    optimizer = optim.Adam(dnc.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98])
    last_save_losses = []
    loss_list = []
    for t in range(1000):
        optimizer.zero_grad()
        chx, mhx, last_read = dnc.init_hidden(None, True)
        seq_length = 30
        input_data, target_output = generate_data(args.batch_size, seq_length, 20)
        input_data = input_data.transpose(1, 0, 2)
        target_output = target_output.transpose(1, 0, 2)
        target_output = target_output[seq_length+1:,:,:]
        target_output = torch.tensor(target_output, requires_grad=True)
        output_list = []
        for i in range(len(input_data)):
            input_tensor = torch.tensor(input_data[i], requires_grad=True)
            output_tensor, (chx, mhx, last_read) = dnc(input_tensor, hx=(chx, mhx, last_read))
            if i > seq_length:
                output_list.append(output_tensor)
        outputs = torch.stack(output_list)
        loss = criterion((outputs), target_output)
        mhx = { k : (v.detach() if v.requires_grad else v) for k, v in mhx.items() }
        torch.nn.utils.clip_grad_norm(dnc.parameters(), args.clip_value)
        optimizer.step()
        loss_value = loss.data[0]
        last_save_losses.append(loss_value)
        loss_list.append(loss_value)
        summarize = (t % 10 == 0)
        if summarize:
            loss = np.mean(last_save_losses)
            print(t, "Avg. Logistic Loss: %.4f" % (loss))
            last_save_losses = []


if __name__ == "__main__":
    test_dnc()
