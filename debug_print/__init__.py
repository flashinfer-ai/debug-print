import torch
from ._kernels import print_tensor as print_tensor_kernel


def print_tensor(x: torch.Tensor, print_ptr: bool = False):
    print_tensor_kernel(x, print_ptr)
