import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, matmul
from torch.nn import Unfold

class ComputeI:

    @classmethod
    def compute_cov_a(cls, a, module):
        return cls.__call__(a, module)

    @classmethod
    def __call__(cls, a, module):
        if isinstance(module, nn.Linear):
            II, I = cls.linear(a, module)
            return II, I
        elif isinstance(module, nn.Conv2d):
            II, I = cls.conv2d(a, module)
            return II, I
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            return None

    @staticmethod
    def conv2d(input, module):
        f = Unfold(
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )
        I = f(input)
        N = I.shape[0]
        K = I.shape[1]
        L = I.shape[2]
        M = module.out_channels
        module.param_shapes = [N, K, L, M]
        if (L*L) * (K + M) < K * M :
            II = einsum("nkl,qkp->nqlp", (I, I))
            module.optimized = True
            return II, I
        else:
            module.optimized = False
            return None, I

    @staticmethod
    def linear(input, module):
        I = input        
        II =  einsum("ni,li->nl", (I, I))   
        module.optimized = True
        return II, I

class ComputeG:

    @classmethod
    def compute_cov_g(cls, g, module):
        """
        :param g: gradient
        :param module: the corresponding module
        :return:
        """
        return cls.__call__(g, module)

    @classmethod
    def __call__(cls, g, module):
        if isinstance(module, nn.Conv2d):
            GG, G = cls.conv2d(g, module)
            return GG, G
        elif isinstance(module, nn.Linear):
            GG, G = cls.linear(g, module)
            return GG, G
        else:
            return None
        

    @staticmethod
    def conv2d(g, module):
        n = g.shape[0]
        g_out_sc = n * g
        grad_output_viewed = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)
        G = grad_output_viewed

        N = module.param_shapes[0]
        K = module.param_shapes[1]
        L = module.param_shapes[2]
        M = module.param_shapes[3]
        if (L*L) * (K + M) < K * M :
            GG = einsum("nml,qmp->nqlp", (G, G))
            module.optimized = True
            return GG, G
        else:
             module.optimized = False
             return None, G

    @staticmethod
    def linear(g, module):
        n = g.shape[0]
        g_out_sc = n * g
        G = g_out_sc
        GG =  einsum("no,lo->nl", (G, G))
        module.optimized = True
        return GG, G
