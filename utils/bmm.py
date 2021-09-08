import math
from torch import nn
from torch.autograd import Function
import torch
import ctypes

import bmm_cuda

torch.manual_seed(42)


class BMM():
    def __init__(self, A, B, C, batch_size, offset_A, offset_B, offset_C, m_magma, n_magma, k_magma):

        self.mul_op = bmm_cuda.BatchMatmul()
        self.mul_op.set_pointers(A, B, C, batch_size, offset_A, offset_B, offset_C)
        self.mul_op.make_streams(batch_size)
        self.mul_op.magma_initialize(m_magma, n_magma, k_magma)

    def MagmaForward(self, A, B, C, m , n, k, batch_size):
        return self.mul_op.MagmaForward(m, n, k, batch_size)

    def CublasForward(self, A, B, C, m , n, k, batch_size, offset_A, offset_B, offset_C, A_T=False, B_T=False):
        return self.mul_op.CublasForward(A, B, C, m, n, k, batch_size, offset_A, offset_B, offset_C, A_T, B_T)
