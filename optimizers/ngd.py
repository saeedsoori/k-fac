import math

import torch
import torch.optim as optim

from utils.ngd_utils import (ComputeI, ComputeG)
from torch import einsum, eye, matmul, cumsum
# from torch.linalg import inv, svd

class NGDOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.01,
                 momentum=0.9,
                 damping=0.1,
                 kl_clip=0.01,
                 weight_decay=0.003,
                 freq=100,
                 gamma=0.9,
                 low_rank='true',
                 super_opt='false',
                 reduce_sum='false',
                 diag='false',
                 rand_svd = False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
       
        super(NGDOptimizer, self).__init__(model.parameters(), defaults)
        
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        # self.grad_outputs = {}
        self.IHandler = ComputeI()
        self.GHandler = ComputeG()
        self.model = model
        self.rank_all={}
        self.rand_svd = rand_svd

        self.S_r = {}
        self.V_r = {}
        self._prepare_model()

        self.steps = 0
        self.m_I = {}
        self.m_G = {}
        self.m_UV = {}
        
        self.m_NGD_Kernel = {}
        self.m_bias_Kernel = {}

        self.kl_clip = kl_clip
        self.freq = freq
        self.gamma = gamma
        self.low_rank = low_rank
        self.super_opt = super_opt
        self.reduce_sum = reduce_sum
        self.diag = diag
        self.damping = damping

    def _save_input(self, module, input):
        # storing the optimized input in forward pass
        if torch.is_grad_enabled() and self.steps % self.freq == 0:
            II, I = self.IHandler(input[0].data, module, self.super_opt, self.reduce_sum, self.diag)
            self.m_I[module] = II, I

    def _save_grad_output(self, module, grad_input, grad_output):
        # storing the optimized gradients in backward pass
        if self.acc_stats and self.steps % self.freq == 0:
            GG, G = self.GHandler(grad_output[0].data, module, self.super_opt, self.reduce_sum, self.diag)
            self.m_G[module] = GG, G

    def _prepare_model(self):
        count = 0
        print(self.model)
        print('NGD keeps the following modules:')
        for module in self.model.modules():
            self.rank_all[module] = []
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1
                
    # Define randomized SVD function
    def rSVD(self, X,r,q,p):
        # Step 1: Sample column space of X with P matrix
        ny = X.shape[1]
        P = torch.randn(ny,r+p).to(X.device)
        Z = X @ P
        for k in range(q):
            Z = X @ (X.T @ Z)
    
        Q, R = torch.linalg.qr(Z,mode='reduced')
        # Step 2: Compute SVD on projected Y = Q.T @ X
        Y = Q.T @ X
        UY, S, VT = torch.linalg.svd(Y,full_matrices=False)
        # note: should reconstruct U from UY in the original algorithm
        # with U = Q @ UY. Here, since X is symmetry, U = V,
        # so we can reuse V and disregard U

        return None, S, VT

    def _update_inv(self, m):
        classname = m.__class__.__name__.lower()
        if classname == 'linear':
            assert(m.optimized == True)
            II = self.m_I[m][0]
            GG = self.m_G[m][0]
            n = II.shape[0]

            ### bias kernel is GG (II = all ones)
            bias_kernel = GG / n
            bias_inv = torch.inverse(bias_kernel + self.damping * eye(n).to(GG.device))
            self.m_bias_Kernel[m] = bias_inv

            NGD_kernel = (II * GG) / n
            if self.rand_svd:
                # U, S, Vh = torch.linalg.svd(NGD_kernel, full_matrices=False)
                U, S, Vh = self.rSVD(NGD_kernel, 50, 0, 20)
                # cs = torch.cumsum(S, dim=0)
                # cs_norm = cs / torch.sum(S)
                self.S_r[m] = inv(torch.diag(S) + self.damping * eye(S.shape[0]).to(II.device))
                self.V_r[m] =  Vh
            else:
                NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))

            if not self.rand_svd:
                self.m_NGD_Kernel[m] = NGD_inv

            self.m_I[m] = (None, self.m_I[m][1])
            self.m_G[m] = (None, self.m_G[m][1])
            torch.cuda.empty_cache()
        elif classname == 'conv2d':
            # SAEED: @TODO: we don't need II and GG after computations, clear the memory
            if m.optimized == True:
                # print('=== optimized ===')
                II = self.m_I[m][0]
                GG = self.m_G[m][0]
                n = II.shape[0]

                NGD_kernel = None
                if self.reduce_sum == 'true':
                    if self.diag == 'true':
                        NGD_kernel = (II * GG / n)
                        NGD_inv = torch.reciprocal(NGD_kernel + self.damping)
                    else:
                        NGD_kernel = II * GG / n
                        if self.rand_svd:
                            _, S, Vh = self.rSVD(NGD_kernel, 50, 0, 20)
                            self.S_r[m] = torch.inverse(torch.diag(S) + self.damping * eye(S.shape[0]).to(II.device))
                            self.V_r[m] =  Vh
                        else:
                            NGD_inv = torch.inverse(NGD_kernel + self.damping * eye(n).to(II.device))
                        
                else:
                    NGD_kernel = (einsum('nqlp->nq', II * GG)) / n
                    NGD_inv = torch.inverse(NGD_kernel + self.damping * eye(n).to(II.device))

                if not self.rand_svd:
                    self.m_NGD_Kernel[m] = NGD_inv

                self.m_I[m] = (None, self.m_I[m][1])
                self.m_G[m] = (None, self.m_G[m][1])
                torch.cuda.empty_cache()
            else:
                # SAEED: @TODO memory cleanup
                I = self.m_I[m][1]
                G = self.m_G[m][1]
                n = I.shape[0]
                AX = einsum("nkl,nml->nkm", (I, G))

                del I
                del G

                AX_ = AX.reshape(n , -1)
                out = matmul(AX_, AX_.t())

                del AX

                NGD_kernel = out / n
                ### low-rank approximation of Jacobian
                if self.low_rank == 'true':
                    # print('=== low rank ===')
                    V, S, U = torch.svd(AX_.T, full_matrices=False)
                    U = U.t()
                    V = V.t()
                    cs = cumsum(S, dim = 0)
                    sum_s = sum(S)
                    index = ((cs - self.gamma * sum_s) <= 0).sum()
                    U = U[:, 0:index]
                    S = S[0:index]
                    V = V[0:index, :]
                    self.m_UV[m] = U, S, V

                del AX_

                NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(NGD_kernel.device))
                self.m_NGD_Kernel[m] = NGD_inv

                del NGD_inv
                self.m_I[m] = None, self.m_I[m][1]
                self.m_G[m] = None, self.m_G[m][1]
                torch.cuda.empty_cache()
    
    
    
    def _get_natural_grad(self, m, damping):
        grad = m.weight.grad.data
        classname = m.__class__.__name__.lower()

        if classname == 'linear':
            assert(m.optimized == True)
            I = self.m_I[m][1]
            G = self.m_G[m][1]
            n = I.shape[0]
            if not self.rand_svd:
                NGD_inv = self.m_NGD_Kernel[m]
            grad_prod = einsum("ni,oi->no", (I, grad))
            grad_prod = einsum("no,no->n", (grad_prod, G))

            # v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
            if self.rand_svd:
                f1 = matmul(self.V_r[m], grad_prod.unsqueeze(1))
                f2 = matmul(self.S_r[m], f1)
                v = matmul(self.V_r[m].t(), f2).squeeze()

            else:
                v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

            gv = einsum("n,no->no", (v, G))
            gv = einsum("no,ni->oi", (gv, I))
            gv = gv / n

            bias_update = None
            if m.bias is not None:
                grad_bias = m.bias.grad.data
                if self.steps % self.freq == 0:
                    grad_prod_bias = einsum("o,no->n", (grad_bias, G))
                    v = matmul(self.m_bias_Kernel[m], grad_prod_bias.unsqueeze(1)).squeeze()
                    gv_bias = einsum('n,no->o', (v, G))
                    gv_bias = gv_bias / n
                    bias_update = (grad_bias - gv_bias) / damping
                else:
                    bias_update = grad_bias

            updates = (grad - gv)/damping, bias_update

        elif classname == 'conv2d':
            grad_reshape = grad.reshape(grad.shape[0], -1)
            if m.optimized == True:
                # print('=== optimized ===')
                I = self.m_I[m][1]
                G = self.m_G[m][1]
                n = I.shape[0]
                if not self.rand_svd:
                    NGD_inv = self.m_NGD_Kernel[m]

                if self.reduce_sum == 'true':
                    x1 = einsum("nk,mk->nm", (I, grad_reshape))
                    grad_prod = einsum("nm,nm->n", (x1, G))

                    if self.diag == 'true':
                        v = NGD_inv * grad_prod
                    else:
                        if self.rand_svd:
                            f1 = matmul(self.V_r[m], grad_prod.unsqueeze(1))
                            f2 = matmul(self.S_r[m], f1)
                            v = matmul(self.V_r[m].t(), f2).squeeze()
                        else:
                            v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

                    gv = einsum("n,nm->nm", (v, G))
                    gv = einsum("nm,nk->mk", (gv, I))
                else:
                    x1 = einsum("nkl,mk->nml", (I, grad_reshape))
                    grad_prod = einsum("nml,nml->n", (x1, G))
                    v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                    gv = einsum("n,nml->nml", (v, G))
                    gv = einsum("nml,nkl->mk", (gv, I))
                gv = gv.view_as(grad)
                gv = gv / n

                bias_update = None
                if m.bias is not None:
                    bias_update = m.bias.grad.data
                
                updates = (grad - gv)/damping, bias_update

            else:
                # TODO(bmu): fix low rank
                if self.low_rank.lower() == 'true':
                    # print("=== low rank ===")

                    ###### using low rank structure
                    U, S, V = self.m_UV[m]
                    NGD_inv = self.m_NGD_Kernel[m]
                    G = self.m_G[m][1]
                    n = NGD_inv.shape[0]

                    grad_prod = V @ grad_reshape.t().reshape(-1, 1)
                    grad_prod = torch.diag(S) @ grad_prod
                    grad_prod = U @ grad_prod
                    grad_prod = grad_prod.squeeze()

                    bias_update = None
                    if m.bias is not None:
                        bias_update = m.bias.grad.data

                    v = matmul(NGD_inv, (grad_prod).unsqueeze(1)).squeeze()

                    gv = U.t() @ v.unsqueeze(1)
                    gv = torch.diag(S) @ gv
                    gv = V.t() @ gv

                    gv = gv.reshape(grad_reshape.shape[1], grad_reshape.shape[0]).t()
                    gv = gv.view_as(grad)
                    gv = gv / n

                    updates = (grad - gv)/damping, bias_update
                else:
                    I = self.m_I[m][1]
                    G = self.m_G[m][1]
                    AX = einsum('nkl,nml->nkm', (I, G))

                    del I
                    del G

                    n = AX.shape[0]

                    NGD_inv = self.m_NGD_Kernel[m]

                    grad_prod = einsum('nkm,mk->n', (AX, grad_reshape))
                    v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                    gv = einsum('nkm,n->mk', (AX, v))
                    gv = gv.view_as(grad)
                    gv = gv / n

                    bias_update = None
                    if m.bias is not None:
                        bias_update = m.bias.grad.data

                    updates = (grad - gv) / damping, bias_update

                    del AX
                    del NGD_inv
                    torch.cuda.empty_cache()

        return updates


    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip

        # vg_sum = 0

        # for m in self.model.modules():
        #     classname = m.__class__.__name__
        #     if classname in self.known_modules:
        #         v = updates[m]
        #         vg_sum += (v[0] * m.weight.grad.data).sum().item()
        #         if m.bias is not None:
        #             vg_sum += (v[1] * m.bias.grad.data).sum().item()
        #     elif classname in ['BatchNorm1d', 'BatchNorm2d']:
        #         vg_sum += (m.weight.grad.data * m.weight.grad.data).sum().item()
        #         if m.bias is not None:
        #             vg_sum += (m.bias.grad.data * m.bias.grad.data).sum().item()

        # vg_sum = vg_sum * (lr ** 2)

        # nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.model.modules():
            if m.__class__.__name__ in ['Linear', 'Conv2d']:
                v = updates[m]
                m.weight.grad.data.copy_(v[0])
                # m.weight.grad.data.mul_(nu)
                if v[1] is not None:
                    m.bias.grad.data.copy_(v[1])
                    # m.bias.grad.data.mul_(nu)
            # elif m.__class__.__name__ in ['BatchNorm1d', 'BatchNorm2d']:
            #     m.weight.grad.data.mul_(nu)
            #     if m.bias is not None:
            #         m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                # print('=== step ===')
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # if weight_decay != 0 and self.steps >= 10 * self.freq:
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p.copy_(buf)

                p.data.add_(-group['lr'], d_p)
                # print('d_p:', d_p.shape)
                # print(d_p)

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.freq == 0:
                self._update_inv(m)
            v = self._get_natural_grad(m, damping)
            updates[m] = v
        
        # if self.steps % self.freq == 0:
        #     for m in self.modules:
        #         print(m)
        #     for m in self.modules:
        #         print(self.rank_all[m])
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
