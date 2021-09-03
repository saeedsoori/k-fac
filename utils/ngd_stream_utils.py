import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum, matmul
from torch.nn import Unfold

class ComputeI:

    @classmethod
    def compute_cov_a(cls, a, module, super_opt='false', reduce_sum='false', diag='false'):
        return cls.__call__(a, module, super_opt, reduce_sum, diag)

    @classmethod
    def __call__(cls, a, module, super_opt='false', reduce_sum='false', diag='false'):
        if isinstance(module, nn.Linear):
            II, I = cls.linear(a, module, super_opt, reduce_sum, diag)
            return II, I, [], []
        elif isinstance(module, nn.Conv2d):
            II, I, A, E = cls.conv2d(a, module, super_opt, reduce_sum, diag)
            return II, I, A, E
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            return None

    @staticmethod
    def conv2d(input, module, super_opt='false', reduce_sum='false', diag='false'):
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

        
        # r = torch.reshape(sum_x, (-1,))
        # r = sum_x.repeat_interleave(9)
        # print(r.shape)
        # I_rs_E = I_rs - r.view_as(I_rs)
        # print('Input I shapes [N K L M]:', module.param_shapes)
        # print('zart'*1000)

        if reduce_sum == 'true':
            I = einsum("nkl->nk", I)
            sum_x = torch.sum(input, dim=(-2,-1))

            r = sum_x.repeat_interleave(module.kernel_size[0]*module.kernel_size[1])
            A = r.view_as(I) 
            E = I - A

            u,s,v = torch.linalg.svd(E, full_matrices=False)
            # cs = torch.cumsum(s, dim=0)/torch.sum(s)
            rank = 10
            U = u[:, 0:rank]
            S = s[0:rank]
            V = torch.diag(S) @ v[0:rank,:]
            print('U S V:', U.shape, S.shape, V.shape)

            E_estim = torch.matmul(U,V)
            print(E_estim.shape)
            print(torch.norm(E - E_estim)/torch.norm(E))
            if diag == 'true':
                I /= L
                II = torch.sum(I * I, dim=1)
            else:
                II = einsum("nk,qk->nq", (I, I))
            module.optimized = True
            return II, I, sum_x, E
            # return II, torch.reshape(I, [1,-1])

        

    @staticmethod
    def linear(input, module, super_opt='false', reduce_sum='false', diag='false'):
        I = input        
        II =  einsum("ni,li->nl", (I, I))   
        module.optimized = True
        return II, I

class ComputeG:

    @classmethod
    def compute_cov_g(cls, g, module, super_opt='false', reduce_sum='false', diag='false'):
        """
        :param g: gradient
        :param module: the corresponding module
        :return:
        """
        return cls.__call__(g, module, super_opt, reduce_sum, diag)

    @classmethod
    def __call__(cls, g, module, super_opt='false', reduce_sum='false', diag='false'):
        if isinstance(module, nn.Conv2d):
            GG, G = cls.conv2d(g, module, super_opt, reduce_sum, diag)
            return GG, G
        elif isinstance(module, nn.Linear):
            GG, G = cls.linear(g, module, super_opt, reduce_sum, diag)
            return GG, G
        else:
            return None
        

    @staticmethod
    def conv2d(g, module, super_opt='false', reduce_sum='false', diag='false'):
        n = g.shape[0]
        g_out_sc = n * g
        grad_output_viewed = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)
        G = grad_output_viewed

        N = module.param_shapes[0]
        K = module.param_shapes[1]
        L = module.param_shapes[2]
        M = module.param_shapes[3]

        if reduce_sum == 'true':
            G = einsum("nkl->nk", G)
            if diag == 'true':
                G /= L
                GG = torch.sum(G * G, dim=1)
            else:
                GG = einsum("nk,qk->nq", (G, G))
            module.optimized = True
            return GG, G

        

    @staticmethod
    def linear(g, module, super_opt='false', reduce_sum='false', diag='false'):
        n = g.shape[0]
        g_out_sc = n * g
        G = g_out_sc
        GG =  einsum("no,lo->nl", (G, G))
        module.optimized = True
        return GG, G
