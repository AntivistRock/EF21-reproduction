#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import mlflow
from math import ceil
import time
from torch.multiprocessing import Process
from torch.autograd import Variable

from optimizer.ef import EF
from optimizer.ef21 import EF21
from optimizer.ef21plus import EF21Plus
from data_utils.utils import partition_dataset
from config import experiment_config


class Net(nn.Module):
    """ Logistic regression"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(112, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


def average_gradients(model):
    r""" Computation of the \nabla f(x) = 1/n \sum \nabla f_i(x)"""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if type(param) is torch.nn.parameter.Parameter:
            dist.reduce(param.grad.data, dst=0)
            param.grad.data /= size


def run(rank, size):
    torch.manual_seed(1234)
    lr_config = {'top1': {'mushrooms': 0.002259, 'a9a': 0.003239}}

    dataset = experiment_config['dataset']
    lr_mult = experiment_config['lr_mult']
    k = experiment_config['k']
    
    lr = lr_config[f'top{k}'][dataset] * lr_mult

    if dist.get_rank() == 0:
        mlflow.set_experiment(f"EF-stepsize-tolerance")
        mlflow.start_run(run_name=f'EF:k={k}; {lr_mult}X')
        hyperparams = {
            "k": 1,
            "dataset": dataset,
            "X": lr_mult,
        }
        mlflow.log_params(hyperparams)

    train_set, bsz = partition_dataset()
    model = Net()
    model = model
#    model = model.cuda(rank)
    optimizer = EF(model.parameters(), lr=lr, k=k)

    for epoch in range(1300):
        if rank == 0:
            optimizer.step()
            for param in model.parameters():
                if type(param) == torch.nn.parameter.Parameter:
                    param.grad = torch.zeros_like(param.data)
        else:
            epoch_loss = 0.0
            for data, target in train_set:
                optimizer.zero_grad()
                data, target = Variable(data), Variable(target)
    #           data, target = Variable(data.cuda(rank)), Variable(target.cuda(rank))
                output = model(data)
                loss = F.binary_cross_entropy_with_logits(output, target)
                epoch_loss += loss
                loss.backward()
                optimizer.step()
        average_gradients(model)

        if dist.get_rank() == 0:
            norm = 0
            for param in model.parameters():
                if type(param) is torch.nn.parameter.Parameter:
                    norm += param.grad.flatten().square().sum()
            mlflow.log_metric('grad-norm', norm, step=epoch*32*k)

    if dist.get_rank() == 0:
        mlflow.end_run()


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29506'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 20
    for lr_mult in [1, 8, 16, 32, 64, 128]:
        experiment_config['lr_mult'] = lr_mult
        
        processes = []
        for rank in range(size):
            p = Process(target=init_processes, args=(rank, size, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
