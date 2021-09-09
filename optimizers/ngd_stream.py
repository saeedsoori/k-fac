import math

import torch
import torch.optim as optim

from utils.ngd_stream_utils import (ComputeI, ComputeG)
from torch import einsum, eye, matmul, cumsum
from torch.linalg import inv, svd
from utils.bmm import BMM

import time
class NGDStreamOptimizer(optim.Optimizer):
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
                 perturb='false',
                 diag='false'):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
       
        super(NGDStreamOptimizer, self).__init__(model.parameters(), defaults)
        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        # self.grad_outputs = {}
        self.IHandler = ComputeI()
        self.GHandler = ComputeG()
        self.model = model
        self.first_iter = {}
        self.dims = {}
        self.is_conv2d = {}
        self.index = {}
        self.count = -1

        # mxk kxn > mxn
        self.mshapes = []
        self.nshapes = []
        self.kshapes = []
        self.cinshapes = []
        self.perturb = perturb
        self._prepare_model()
        # print('H'*100)
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


        

    def initialize(self):
        # print('initialize'*100)
        ## allocate memory for matrix I
        # device = torch.device("cuda")
        device = torch.device("cuda")
        dtype = torch.float32
        kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

        # I_size = 0
        G_size = 0
        grad_size = 0
        grad_rs_size = 0
        A_size = 0

        # I_offset = [0]
        A_offset = [0]
        G_offset = [0]
        grad_offset = [0]
        grad_rs_offset = [0]
        i = 0
        for v in self.dims:
            n = self.dims[v][0]
            k = self.dims[v][1]
            m = self.dims[v][-1]

            if self.is_conv2d[v]:
                cin = self.dims[v][2]
                print('cin for conv:', cin)
                A_size += n * cin
            else:
                A_size += n * k
                cin = k
                print('cin for linear:', cin)



            # I_size += n * k
            G_size += n * m
            grad_size += k * m
            grad_rs_size += cin * m
            
            # I_offset.append(I_size)
            G_offset.append(G_size)
            grad_offset.append(grad_size)
            grad_rs_offset.append(grad_rs_size)
            A_offset.append(A_size)
            self.index[v] = i
            self.mshapes.append(n)
            self.nshapes.append(m)
            self.kshapes.append(k)
            self.cinshapes.append(cin)
            i += 1

        self.mshapes.append(0)
        self.nshapes.append(0)
        self.kshapes.append(0)
        self.cinshapes.append(0)


        self.m_magma = torch.IntTensor(self.mshapes)
        self.n_magma = torch.IntTensor(self.nshapes)
        self.k_magma = torch.IntTensor(self.kshapes)

        # self.m_magma = torch.cuda.IntTensor(self.mshapes)
        # self.n_magma = torch.cuda.IntTensor(self.nshapes)
        # self.k_magma = torch.cuda.IntTensor(self.kshapes)

        self.m_arr = self.mshapes
        self.n_arr = self.nshapes
        self.k_arr = self.kshapes
        self.cin_arr = self.cinshapes

        # self.I_MEM = torch.zeros(I_size, **kwargs)
        self.G_MEM = torch.zeros(G_size, **kwargs)
        self.grad_MEM = torch.zeros(grad_size, **kwargs)
        self.grad_rs_MEM = torch.zeros(grad_rs_size, **kwargs)
        self.grad_gv_MEM = torch.zeros(grad_size, **kwargs)
        self.grad_gv_rs_MEM = torch.zeros(grad_rs_size, **kwargs)
        self.grad_prod_MEM = torch.zeros(G_size, **kwargs)
        self.gv_MEM = torch.zeros(G_size, **kwargs)
        self.gv_temp_MEM = torch.zeros(G_size, **kwargs)
        self.A_MEM = torch.zeros(A_size, **kwargs)
        self.NGD_inv_MEM = torch.zeros((self.mshapes[0], self.mshapes[0] * self.count) , **kwargs)
        self.A_offset = A_offset
        # self.I_offset = I_offset
        self.G_offset = G_offset
        self.grad_offset = grad_offset
        self.grad_rs_offset = grad_rs_offset

        # self.Mul = BMM(self.I_MEM, self.grad_MEM, self.grad_prod_MEM, self.count, self.I_offset, self.grad_offset, self.G_offset, self.m_magma, self.n_magma, self.k_magma)
        self.Mul = BMM(self.A_MEM, self.grad_rs_MEM, self.grad_prod_MEM, self.count, self.A_offset, self.grad_rs_offset, self.G_offset, self.m_magma, self.n_magma, self.k_magma)
        # self.Mul2 = BMM(self.G_MEM, self.A_MEM, self.grad_gv_rs_MEM, self.count, self.G_offset, self.A_offset, self.grad_rs_offset, self.m_magma, self.n_magma, self.k_magma)

        print('Initialization finished')


    def _save_input(self, module, input):
        # storing the optimized input in forward pass

        if torch.is_grad_enabled() and self.steps % self.freq == 0 and self.first_iter[module] == False:
            II, I, A, E, U, V = self.IHandler(input[0].data, module, self.super_opt, self.reduce_sum, self.diag, self.perturb)
            self.m_I[module] = II, I, A, E, U, V
            # print('module:', module)
            # print(self.m_I[module][2])
            index = self.index[module]
            # st = self.I_offset[index]
            # end = self.I_offset[index+1]
            # self.I_MEM[st:end] = torch.reshape(I, [1,-1])
            st = self.A_offset[index]
            end = self.A_offset[index+1]
            self.A_MEM[st:end] = torch.reshape(A, [1,-1])

        if self.first_iter[module] == True:
            if module.__class__.__name__ == 'Conv2d':
                # print(self.steps)
                # N K L M
                N = input[0].shape[0]
                K = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
                L = input[0].shape[-2] * input[0].shape[-1]
                M = module.weight.shape[0]
                ## verify next line
                Cin = module.weight.shape[1] 
                print('Cin:', Cin)
                self.dims[module] = [N, K, Cin, M]
                # print('Computed shapes:', self.dims[module])
                self.first_iter[module] = False
                self.is_conv2d[module] = True

            elif module.__class__.__name__ == 'Linear':
                # TODO: add computations for linear layer too
                N = input[0].shape[0]
                K = input[0].shape[1]
                M = module.weight.shape[0]
                cin = 0
                self.dims[module] = [N, K, cin, M]
                self.is_conv2d[module] = False

                # print('XXXX:', self.dims[module])
                self.first_iter[module] = False


        

    def _save_grad_output(self, module, grad_input, grad_output):
        # storing the optimized gradients in backward pass
        if self.acc_stats and self.steps % self.freq == 0:
            GG, G = self.GHandler(grad_output[0].data, module, self.super_opt, self.reduce_sum, self.diag, self.perturb)
            self.m_G[module] = GG, G
            index = self.index[module]
            st = self.G_offset[index]
            end = self.G_offset[index+1]
            self.G_MEM[st:end] = torch.reshape(G, [1,-1])

            # grad_ = torch.sum(grad, (-2,-1))
            # # print('grad_shape:', grad_.shape)
            # grad_reshape_rs = grad_.reshape(grad_.shape[0], -1)

    def _prepare_model(self):
        count = 0
        print(self.model)
        print('NGD keeps the following modules:')
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.first_iter[module] = True
                self.dims[module] = []
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1
        self.count = count

    def _update_inv(self, m):
        classname = m.__class__.__name__.lower()
        if classname == 'linear':
            assert(m.optimized == True)
            II = self.m_I[m][0]
            GG = self.m_G[m][0]
            n = II.shape[0]

            ### bias kernel is GG (II = all ones)
            bias_kernel = GG / n
            bias_inv = inv(bias_kernel + self.damping * eye(n).to(GG.device))
            self.m_bias_Kernel[m] = bias_inv

            NGD_kernel = (II * GG) / n
            NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))
            self.m_NGD_Kernel[m] = NGD_inv

            self.m_I[m] = None, self.m_I[m][1], self.m_I[m][2], self.m_I[m][3], self.m_I[m][4], self.m_I[m][5]
            self.m_G[m] = None, self.m_G[m][1]
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
                        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))
                else:
                    NGD_kernel = (einsum('nqlp->nq', II * GG)) / n
                    NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(II.device))

                self.m_NGD_Kernel[m] = NGD_inv

                self.m_I[m] = None, self.m_I[m][1], self.m_I[m][2], self.m_I[m][3], self.m_I[m][4], self.m_I[m][5]
                self.m_G[m] = None, self.m_G[m][1]
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
                    V, S, U = svd(AX_.T, full_matrices=False)
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
                self.m_I[m] = None, self.m_I[m][1], self.m_I[m][2], self.m_I[m][3], self.m_I[m][4], self.m_I[m][5]
                self.m_G[m] = None, self.m_G[m][1]
                torch.cuda.empty_cache()
    def _get_natural_grad_struct_all(self, updates, damping):
        a = 2
        # grad = m.weight.grad.data
        # classname = m.__class__.__name__.lower()




        # result = self.Mul.CublasForward(self.A_MEM, self.grad_rs_MEM, self.grad_prod_MEM, self.m_arr, self.n_arr, self.cin_arr, self.count, self.A_offset, self.grad_rs_offset, self.G_offset)
        #### doing elementwise  multiplicatio with G
        # self.grad_prod_MEM = self.grad_prod_MEM * self.G_MEM
        ## TODO: SAEED fix the next line by a for loop over modules and reduction seperately
        # print('DD: ', self.G_MEM.shape)
        # print('self.NGD_inv_MEM: ', self.NGD_inv_MEM.shape)
        # self.grad_prod_MEM2 = torch.randn(self.NGD_inv_MEM.shape[1], self.grad_prod_MEM.numel()//self.NGD_inv_MEM.shape[1] )
        # self.grad_prod_MEM3 = torch.sum(self.grad_prod_MEM2, -1)
        
        # v = matmul(self.NGD_inv_MEM, self.grad_prod_MEM3).squeeze()
        # self.G_scaled = self.G_MEM * self.G_MEM

        # AG = torch.matmul(G_scaled.t(), A)
        # result = self.Mul.CublasForward(self.G_scaled, self.A_MEM, self.grad_prod_MEM, self.m_arr, self.cin_arr, self.n_arr, self.count, self.G_offset, self.A_offset, self.G_offset, A_T = True, B_T = False)
        # result = self.Mul.CublasForward(self.A_MEM, self.grad_rs_MEM, self.grad_prod_MEM, self.m_arr, self.n_arr, self.cin_arr, self.count, self.A_offset, self.grad_rs_offset, self.G_offset)

        # AG_expand = torch.repeat_interleave(self.grad_prod_MEM, 9, dim=0)


        #### doing elementwise  multiplicatio with G
        # self.grad_prod_MEM = self.grad_prod_MEM * self.G_MEM
        # for m in self.modules:
        #     grad = m.weight.grad.data
        #     classname = m.__class__.__name__.lower()
        #     if classname == 'linear':
        #         assert(m.optimized == True)
        #         I = self.m_I[m][1]
        #         G = self.m_G[m][1]
        #         n = I.shape[0]
        #         NGD_inv = self.m_NGD_Kernel[m]
        #         grad_prod = einsum("ni,oi->no", (I, grad))
        #         grad_prod = einsum("no,no->n", (grad_prod, G))

        #         v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

        #         gv = einsum("n,no->no", (v, G))
        #         gv = einsum("no,ni->oi", (gv, I))
        #         gv = gv / n

        #         bias_update = None
        #         if m.bias is not None:
        #             grad_bias = m.bias.grad.data
        #             if self.steps % self.freq == 0:
        #                 grad_prod_bias = einsum("o,no->n", (grad_bias, G))
        #                 v = matmul(self.m_bias_Kernel[m], grad_prod_bias.unsqueeze(1)).squeeze()
        #                 gv_bias = einsum('n,no->o', (v, G))
        #                 gv_bias = gv_bias / n
        #                 bias_update = (grad_bias - gv_bias) / damping
        #             else:
        #                 bias_update = grad_bias

        #         updates = (grad - gv)/damping, bias_update

        #     elif classname == 'conv2d':
        #         # print('conv2d grad shape:', grad.shape)
        #         grad_ = torch.sum(grad, (-2,-1))
        #         # print('grad_shape:', grad_.shape)
        #         grad_reshape_rs = grad_.reshape(grad_.shape[0], -1)
        #         grad_reshape = grad.reshape(grad.shape[0], -1)
        #         if m.optimized == True:
        #             # print('=== optimized ===')
        #             I = self.m_I[m][1]
        #             G = self.m_G[m][1]
        #             # we have I = A + E
        #             # notice A is not repeated and this is the reduced version
        #             n = G.shape[0]
        #             NGD_inv = self.m_NGD_Kernel[m]

        #             # print('m:', m)
        #             A = self.m_I[m][2]
        #             E = self.m_I[m][3]
        #             U = self.m_I[m][4]
        #             V = self.m_I[m][5]

                    # the new method computation for x1 = I * g ~ A * g_rs
                    # if m.stride[0] == 1:
                    #     # x1_rs = einsum("nk,mk->nm", (A, grad_reshape_rs))
                    #     x1_rs = torch.matmul(A, grad_reshape_rs.t())
                    #     # e1 = einsum("vk,mk->vm", (V, grad_reshape))
                    #     e1 = torch.matmul(V, grad_reshape.t())
                    #     # e2 = einsum("nv,vm->nm", (U, e1))
                    #     e2 = torch.matmul(U, e1)
                    #     x1 = x1_rs + e2
                    #     ## error:
                    #     ## x1_rs_e = x1_rs + e2
                    #     # err = torch.norm(x1 - x1_rs)/torch.norm(x1)
                    #     # err2 = torch.norm(x1 - x1_rs_e)/torch.norm(x1)
                    #     # print('err and err2: ', err, err2)
                    # else:
                    #     x1 = einsum("nk,mk->nm", (I, grad_reshape))


                    # grad_prod = einsum("nm,nm->n", (x1, G))


                    ### we have to replace this operation: gv = einsum("nm,nk->mk", (gv, I))
                    ## I ~ A + U * V
                    # A has dimension (batch, cin)
                    # U and V have dimensions (cin, r) and (r, cout) where r is the rank parameter (default=1).

                    
                    # v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

                    # G_scaled = einsum("n,nm->nm", (v, G))
                    # if m.stride[0] == 1:
                    #     # AG = einsum("nk,nm->mk", (A, G_scaled))
                    #     AG = torch.matmul(G_scaled.t(), A)
                    #     AG_expand = torch.repeat_interleave(AG, m.kernel_size[0] * m.kernel_size[1], dim=1)
                    #     # aux_var = einsum("nr,nm->rm", (U, G_scaled))
                    #     aux_var = torch.matmul(U.t(), G_scaled)
                    #     # residu = einsum("rk,rm->mk", (V, aux_var))
                    #     residu = torch.matmul(aux_var.t(), V)

                    #     gv = AG_expand + residu
                        
                    #     ## Error:
                    #     ## gv_stim = AG_expand + residu
                    #     # err_gv = torch.norm(gv - gv_stim)/torch.norm(gv)
                    #     # print('err_gv: ', err_gv)
                    # else:
                    #     gv = einsum("nm,nk->mk", (G_scaled, I))

                    # gv = gv.view_as(grad)
                    # gv = gv / n

                    # bias_update = None
                    # if m.bias is not None:
                    #     bias_update = m.bias.grad.data
                    
                    # updates = (grad - gv)/damping, bias_update

                
            # return updates

    def _get_natural_grad_struct(self, m, damping):
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
            # print('conv2d grad shape:', grad.shape)
            grad_ = torch.sum(grad, (-2,-1))
            # print('grad_shape:', grad_.shape)
            grad_reshape_rs = grad_.reshape(grad_.shape[0], -1)
            grad_reshape = grad.reshape(grad.shape[0], -1)
            if m.optimized == True:
                # print('=== optimized ===')
                I = self.m_I[m][1]
                G = self.m_G[m][1]
                # we have I = A + E
                # notice A is not repeated and this is the reduced version
                n = G.shape[0]
                NGD_inv = self.m_NGD_Kernel[m]

                if self.reduce_sum == 'true':
                    # print('m:', m)
                    A = self.m_I[m][2]
                    E = self.m_I[m][3]
                    U = self.m_I[m][4]
                    V = self.m_I[m][5]

                    # the new method computation for x1 = I * g ~ A * g_rs
                    if m.stride[0] == 1:
                        # x1_rs = einsum("nk,mk->nm", (A, grad_reshape_rs))
                        x1_rs = torch.matmul(A, grad_reshape_rs.t())
                        # e1 = einsum("vk,mk->vm", (V, grad_reshape))
                        e1 = torch.matmul(V, grad_reshape.t())
                        # e2 = einsum("nv,vm->nm", (U, e1))
                        e2 = torch.matmul(U, e1)
                        x1 = x1_rs + e2
                        ## error:
                        ## x1_rs_e = x1_rs + e2
                        # err = torch.norm(x1 - x1_rs)/torch.norm(x1)
                        # err2 = torch.norm(x1 - x1_rs_e)/torch.norm(x1)
                        # print('err and err2: ', err, err2)
                    else:
                        x1 = einsum("nk,mk->nm", (I, grad_reshape))


                    grad_prod = einsum("nm,nm->n", (x1, G))


                    ### we have to replace this operation: gv = einsum("nm,nk->mk", (gv, I))
                    ## I ~ A + U * V
                    # A has dimension (batch, cin)
                    # U and V have dimensions (cin, r) and (r, cout) where r is the rank parameter (default=1).

                    if self.diag == 'true':
                        v = NGD_inv * grad_prod
                    else:
                        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

                    G_scaled = einsum("n,nm->nm", (v, G))
                    if m.stride[0] == 1:
                        # AG = einsum("nk,nm->mk", (A, G_scaled))
                        AG = torch.matmul(G_scaled.t(), A)
                        AG_expand = torch.repeat_interleave(AG, m.kernel_size[0] * m.kernel_size[1], dim=1)
                        # aux_var = einsum("nr,nm->rm", (U, G_scaled))
                        aux_var = torch.matmul(U.t(), G_scaled)
                        # residu = einsum("rk,rm->mk", (V, aux_var))
                        residu = torch.matmul(aux_var.t(), V)

                        gv = AG_expand + residu
                        
                        ## Error:
                        ## gv_stim = AG_expand + residu
                        # err_gv = torch.norm(gv - gv_stim)/torch.norm(gv)
                        # print('err_gv: ', err_gv)
                    else:
                        gv = einsum("nm,nk->mk", (G_scaled, I))

                    
                    


                    
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
                NGD_inv = self.m_NGD_Kernel[m]

                if self.reduce_sum == 'true':
                    x1 = einsum("nk,mk->nm", (I, grad_reshape))
                    grad_prod = einsum("nm,nm->n", (x1, G))

                    if self.diag == 'true':
                        v = NGD_inv * grad_prod
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

    def _get_natural_grad_all(self, updates, damping):
        result = self.Mul.CublasForward(self.I_MEM, self.grad_MEM, self.grad_prod_MEM, self.m_arr, self.n_arr, self.k_arr, self.count, self.I_offset, self.grad_offset, self.G_offset)

        #### doing elementwise  multiplicatio with G
        self.grad_prod_MEM = self.grad_prod_MEM * self.G_MEM

        for m in self.modules:
            classname = m.__class__.__name__.lower()

            grad = m.weight.grad.data
            index = self.index[m]
            st = self.I_offset[index]
            end = self.I_offset[index+1]
            I = self.I_MEM[st:end]

            st = self.G_offset[index]
            end = self.G_offset[index+1]
            G = self.G_MEM[st:end]


            I = I.view_as(self.m_I[m][1])
            G = G.view_as(self.m_G[m][1])

            # grad_prod has the same size and shape as G
            grad_prod = self.grad_prod_MEM[st:end]



            grad_prod = grad_prod.view_as(G)
            
            if classname == 'linear':
                assert(m.optimized == True)
                # I = I.view_as(self.m_I[m][1])
                # G = G.view_as(self.m_G[m][1])
                n = I.shape[0]
                NGD_inv = self.m_NGD_Kernel[m]

                ## comment next two line since we compute grad_prod all at once
                # grad_prod_org = I @ grad.t()
                # grad_prod = einsum("no,no->n", (grad_prod, G))
                grad_prod = einsum("no->n", (grad_prod))

                v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

                # self.gv_MEM[st:end] = torch.reshape(einsum("n,no->no", (v, G)), [1, -1])
                self.gv_temp_MEM[st:end] = torch.reshape(einsum("n,no->on", (v, G)), [1, -1])

            elif classname == 'conv2d':
                grad_reshape = grad.reshape(grad.shape[0], -1)
                # grad = grad_reshape.t()
                if m.optimized == True:
                    # print('=== optimized ===')
                    
                    n = I.shape[0]
                    NGD_inv = self.m_NGD_Kernel[m]

                    if self.reduce_sum == 'true':
                        # grad_prod = I @ grad_reshape.t()
                        x1 = grad_prod
                        # x1 = einsum("nk,mk->nm", (I, grad_reshape))
                        # grad_prod = einsum("nm,nm->n", (x1, G))
                        grad_prod = einsum("nm->n", (x1))

                        if self.diag == 'true':
                            v = NGD_inv * grad_prod
                        else:
                            v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

                        # print('m n:', G.shape[1], G.shape[0])
                        # self.gv_MEM[st:end] = torch.reshape(einsum("n,nm->nm", (v, G)), [1,-1])
                        self.gv_temp_MEM[st:end] = torch.reshape(einsum("n,nm->mn", (v, G)), [1,-1])

        # result = self.Mul.CublasForward(self.gv_MEM, self.I_MEM, self.grad_gv_MEM, self.k_arr, self.n_arr, self.m_arr, self.count, self.G_offset, self.I_offset, self.grad_offset)
        result = self.Mul.CublasForward(self.gv_temp_MEM, self.I_MEM, self.grad_gv_MEM, self.n_arr, self.k_arr, self.m_arr, self.count, self.G_offset, self.I_offset, self.grad_offset)
        # print('order should be: m n x n k > mxk therefore: m k n')
        # print(self.k_arr, self.n_arr, self.m_arr)

        for m in self.modules:
            classname = m.__class__.__name__.lower()

            grad = m.weight.grad.data
            index = self.index[m]

            st = self.grad_offset[index]
            end = self.grad_offset[index+1]
            gv = self.grad_gv_MEM[st:end]

            st = self.I_offset[index]
            end = self.I_offset[index+1]
            I = self.I_MEM[st:end]

            st = self.G_offset[index]
            end = self.G_offset[index+1]
            G = self.G_MEM[st:end]

            

            I = I.view_as(self.m_I[m][1])
            G = G.view_as(self.m_G[m][1])  

            if classname == 'linear':
                gv = gv.view_as(grad)

                # gv = einsum("no,ni->oi", (self.gv_MEM[st:end].view_as(G), I))
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

                updates_m = (grad - gv)/damping, bias_update

            elif classname == 'conv2d':
                grad_reshape = grad.reshape(grad.shape[0], -1)
                gv = gv.view_as(grad_reshape)

                # gv = einsum("nm,nk->mk", (self.gv_MEM[st:end].view_as(G), I))
                gv = gv.view_as(grad)
                gv = gv / n

                bias_update = None
                if m.bias is not None:
                    bias_update = m.bias.grad.data
                updates_m = (grad - gv)/damping, bias_update
            updates[m] = updates_m
                

            # return updates_m


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
                
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                #         buf.mul_(momentum).add_(d_p)
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1, d_p)
                #     d_p.copy_(buf)

                # if weight_decay != 0 and self.steps >= 10 * self.freq:
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], d_p)
                # print('d_p:', d_p.shape)
                # print(d_p)

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        updates_new = {}
        k = 0
        # torch.cuda.synchronize()
        # st_time = time.time()
        for m in self.modules:
            classname = m.__class__.__name__.lower()
            if self.steps % self.freq == 0:
                self._update_inv(m)
            # v = self._get_natural_grad_struct(m, damping)
            # v = self._get_natural_grad(m, damping)
            # updates[m] = v
            # updates[m] = torch.Tensor(0), torch.Tensor(0)

            index = self.index[m]
            st = self.grad_offset[index]
            end = self.grad_offset[index+1]
            grad = m.weight.grad.data
            if classname == 'conv2d':
                grad = grad.reshape(grad.shape[0], -1)
                grad_rs = torch.sum(grad, (-2,-1))
            self.grad_MEM[st:end] = torch.reshape(grad, [1,-1])
            self.grad_rs_MEM[st:end] = torch.reshape(grad_rs, [1,-1])
            n = self.mshapes[0]
            self.NGD_inv_MEM[0:n, k * n: n * (k + 1)] = self.m_NGD_Kernel[m]
            k += 1

        # torch.cuda.synchronize()
        # print('Normal:', time.time() - st_time)
        # torch.cuda.synchronize()
        # st_time = time.time()

        # self._get_natural_grad_all(updates, damping)
        self._get_natural_grad_struct_all(updates_new, damping)
        # torch.cuda.synchronize()
        # print('Struct:', time.time() - st_time)

        # for m in self.modules:
        #     print(torch.linalg.norm(updates[m][0] - updates_org[m][0]))
        # self._kl_clip_and_update_grad(updates, lr)
        self._step(closure)
        self.steps += 1
