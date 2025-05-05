#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from optimizer.ef21 import EF21

from math import ceil
from random import Random
from sklearn.datasets import load_svmlight_file
from torch.multiprocessing import Process
from torch.autograd import Variable


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Logistic regression"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(112, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


def partition_dataset():
    dataset = "../data/mushrooms.txt"
    
    data = load_svmlight_file(dataset)
    A, b = data[0].toarray(), data[1]
    b = b - 1
    A = torch.tensor(A, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(A, b)

    size = dist.get_world_size()
    bsz = 406
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def average_gradients(model):
    r""" Computation of the \nabla f(x) = 1/n \sum \nabla f_i(x)"""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if type(param) is torch.nn.parameter.Parameter:
            dist.reduce(param.grad.data, dst=0)
            param.grad.data /= size


def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    model = model
#    model = model.cuda(rank)
    optimizer = EF21(model.parameters(), lr=0.1)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(1000):
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
            # print('Rank ',
            #     dist.get_rank(), ', epoch ', epoch, ': ',
            #     epoch_loss / num_batches)
        average_gradients(model)

        if dist.get_rank() == 0:
            norm = 0
            for param in model.parameters():
                if type(param) is torch.nn.parameter.Parameter:
                    norm += param.grad.flatten().square().sum()
            print("Norm:", norm)


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 20
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()