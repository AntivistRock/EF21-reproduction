import numpy as np
import torch
import torch.distributed as dist
from random import Random
from sklearn.datasets import load_svmlight_file


def compute_lipshitz_const(A):
    n = A.shape[0]
    S, _ = torch.linalg.eig(A.T @ A)
    l_max = S.real.max()

    L = 1 / (4 * n) * l_max + 2 * 0.1
    return L


def get_partitioner(dataset, size):
    partition_sizes = [0] + [1.0 / size for _ in range(size - 1)]
    return DataPartitioner(dataset, partition_sizes)


def compute_gamma(dataset, k):
    d = dataset.tensors[0].shape[1]
    partitioner = get_partitioner(dataset, 21)

    # compute L per node
    Ls = []
    for i in range(1, 21):
        idxs = partitioner.use(i).index
        A_i = partitioner.use(i).data[idxs][0]
        const = compute_lipshitz_const(A_i)
        Ls.append(const.item())
    Ls = torch.tensor(Ls)


    L = Ls.mean()
    L_tilde = Ls.square().mean().sqrt()

    alpha = k / d
    theta = 1 - np.sqrt(1 - alpha)
    beta = (1 - alpha) / theta

    gamma_max = 1 / (L + L_tilde * np.sqrt(beta / alpha))
    return gamma_max.item()


def load_mushrooms():
    dataset = "../data/mushrooms.txt"
    data = load_svmlight_file(dataset)
    A, b = data[0].toarray(), data[1]
    b = b - 1
    A = torch.tensor(A, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32).unsqueeze(1)

    return torch.utils.data.TensorDataset(A, b)

def load_a9a():
    dataset = "../data/a9a.txt"
    data = load_svmlight_file(dataset)
    A, b = data[0].toarray(), data[1]
    b = b - 1
    A = torch.tensor(A, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32).unsqueeze(1)

    return torch.utils.data.TensorDataset(A, b)


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


def partition_dataset():
    dataset = load_mushrooms()

    size = dist.get_world_size()
    bsz = 1628
    partition_sizes = [0] + [1.0 / size for _ in range(size - 1)] #  [0.15] + [(1.0 - 0.15) / size for _ in range(size - 1)]  # first part for master node to test
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    if dist.get_rank() == 0:
        train_set = None
    else:
        train_set = torch.utils.data.DataLoader(
            partition, batch_size=bsz, shuffle=True)
    return train_set, bsz
