import math

import torch
import torch.optim as optim

from utils.ngd_utils import (ComputeI, ComputeG)
from torch import einsum, eye, matmul, cumsum
from torch.linalg import inv, svd

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
                 low_rank='true'):
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
        self._prepare_model()

        self.steps = 0
        self.m_I = {}
        self.m_G = {}
        self.m_UV = {}
        self.m_NGD_Kernel = {}

        self.kl_clip = kl_clip
        self.freq = freq
        self.gamma = gamma
        self.low_rank = low_rank
        self.damping = damping

    def _save_input(self, module, input):
        # storing the optimized input in forward pass
        if torch.is_grad_enabled() and self.steps % self.freq == 0:
            II, I = self.IHandler(input[0].data, module)
            self.m_I[module] = II, I

    def _save_grad_output(self, module, grad_input, grad_output):
        # storing the optimized gradients in backward pass
        if self.acc_stats and self.steps % self.freq == 0:
            GG, G = self.GHandler(grad_output[0].data, module)
            self.m_G[module] = GG, G

    def _prepare_model(self):
        count = 0
        print(self.model)
        print('NGD keeps the following modules:')
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        classname = m.__class__.__name__.lower()
        if classname == 'linear':
            assert(m.optimized == True)
            II = self.m_I[m][0]
            GG = self.m_G[m][0]
            n = II.shape[0]

            ### bias kernel is GG (II = all ones)
            NGD_kernel = (II * GG + GG) / n 
            NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))
            self.m_NGD_Kernel[m] = NGD_inv
        elif classname == 'conv2d':
            # SAEED: @TODO: we don't need II and GG after computations, clear the memory
            if m.optimized == True:
                II = self.m_I[m][0]
                GG = self.m_G[m][0]
                n = II.shape[0]
                ### bias kernel:
                bias_kernel = einsum("nqlp->nq", GG)
                NGD_kernel = (einsum('nqlp->nq', II * GG) + bias_kernel)/ n
                NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))
                self.m_NGD_Kernel[m] = NGD_inv
            else:
                # SAEED: @TODO memory cleanup
                I = self.m_I[m][1]
                G = self.m_G[m][1]
                n = I.shape[0]
                AX = einsum("nkl,nml->nkm", (I, G))
                AX_ = AX.reshape(n , -1)
                out = matmul(AX_, AX_.t()) 

                ### bias kernel:
                bias_kernel = einsum("nml,qml->nq", (G, G))
                NGD_kernel = (out + bias_kernel)/ n
                NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(I.device))
                self.m_NGD_Kernel[m] = NGD_inv
                ### low-rank approximation of Jacobian
                if self.low_rank == 'true':
                    V, S, U = svd(AX_.T, compute_uv=True, full_matrices=False)
                    U = U.t()
                    V = V.t()
                    cs = cumsum(S, dim = 0)
                    sum_s = sum(S)
                    index = ((cs - self.gamma * sum_s) <= 0).sum()
                    U = U[:, 0:index]
                    S = S[0:index]
                    V = V[0:index, :]
                    self.m_UV[m] = U, S, V
                

    def _get_natural_grad(self, m, damping):
        grad = m.weight.grad.data
        classname = m.__class__.__name__.lower()

        if classname == 'linear':
            assert(m.optimized == True)
            I = self.m_I[m][1]
            G = self.m_G[m][1]
            n = I.shape[0]
            NGD_inv = self.m_NGD_Kernel[m]
            grad_prod = einsum("ni,oi->no", (I, grad))
            grad_prod = einsum("no,no->n", (grad_prod, G))

            bias_update = None
            if m.bias is not None:
                grad_bias = m.bias.grad.data
                grad_prod_bias = einsum("o,no->n", (grad_bias, G))
            
            v = matmul(NGD_inv, (grad_prod_bias + grad_prod).unsqueeze(1)).squeeze()
            gv = einsum("n,no->no", (v, G))
            if m.bias is not None:
                gv_bias = einsum("no->o", gv)
                gv_bias = gv_bias / n
                bias_update = (grad_bias - gv_bias)/damping
            gv = einsum("no,ni->oi", (gv, I))
            gv = gv / n
            updates = (grad - gv)/damping, bias_update

        elif classname == 'conv2d':
            grad_reshape = grad.reshape(grad.shape[0], -1)
            if m.optimized == True:
                I = self.m_I[m][1]
                G = self.m_G[m][1]
                n = I.shape[0]
                NGD_inv = self.m_NGD_Kernel[m]

                x1 = einsum("nkl,mk->nml", (I, grad_reshape))
                grad_prod = einsum("nml,nml->n", (x1, G)) 

                bias_update = None
                if m.bias is not None:
                    grad_bias = m.bias.grad.data
                    grad_prod_bias = einsum("nml,m->n", (G, grad_bias))

                v = matmul(NGD_inv, (grad_prod + grad_prod_bias).unsqueeze(1)).squeeze()
                gv = einsum("n,nml->nml", (v, G))
                if m.bias is not None:
                    gv_bias = einsum("nml->m", gv)
                    gv_bias = gv_bias.view_as(grad_bias)
                    gv_bias = gv_bias / n
                    bias_update = (grad_bias - gv_bias)/damping

                gv = einsum("nml,nkl->mk", (gv, I))
                gv = gv.view_as(grad)
                gv = gv / n
                
                updates = (grad - gv)/damping, bias_update

            else:
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
                    grad_bias = m.bias.grad.data
                    grad_prod_bias = einsum("nml,m->n", (G, grad_bias))

                v = matmul(NGD_inv, (grad_prod + grad_prod_bias).unsqueeze(1)).squeeze()
                
                if m.bias is not None:
                    gv_bias = einsum("n,nml->m", (v, G))
                    gv_bias = gv_bias.view_as(grad_bias)
                    gv_bias = gv_bias / n
                    bias_update = (grad_bias - gv_bias)/damping

                gv = U.t() @ v.unsqueeze(1)
                gv = torch.diag(S) @ gv
                gv = V.t() @ gv

                gv = gv.reshape(grad_reshape.shape[1], grad_reshape.shape[0]).t()
                gv = gv.view_as(grad)
                gv = gv / n

                updates = (grad - gv)/damping, bias_update
        return updates


    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip

        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                # if weight_decay != 0 and self.steps >= 10 * self.freq:
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], d_p)

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
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
