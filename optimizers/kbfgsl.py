import math
import copy

import torch
import torch.optim as optim

from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat


class KBFGSLOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 weight_decay=0,
                 stat_decay=0.9,
                 damping=0.3,
                 TCov=1,
                 TInv=100,
                 batch_averaged=True,
                 num_s_y_pairs=100):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): optimizer now only support model as input
        super(KBFGSLOptimizer, self).__init__(model.parameters(), defaults)
        self.handles = []
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.pre_activations, self.next_pre_activations = {}, {}

        self.model = model
        self._prepare_model(self.model, cloned=False, init_module=True)

        self.steps = 0

        self.stat_decay = stat_decay
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.damping = damping
        self.lr = lr

        self.TCov = TCov
        self.TInv = TInv

        self.num_s_y_pairs = num_s_y_pairs
        print('num_s_y_pairs:', self.num_s_y_pairs)

        # BFGS specific
        # TODO(bmu): may compute As_a without explicitly computing A
        self.m_aa, self.As = {}, {}

        # a, g averaged over batch + spatial dimension for conv; over batch for fc
        self.a_avg, self.g_avg, self.next_g_avg = {}, {}, {}

        self.s_a, self.y_a, self.H_a = {}, {}, {}
        self.s_g, self.y_g = {}, {}

        self.s, self.y, self.R_inv, self.yTy, self.D_diag = {}, {}, {}, {}, {}

        self.left_matrix, self.right_matrix = {}, {}

        self.gamma = 1

    def _save_input(self, m, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            if m.__class__.__name__ == "Conv2d":
                # KF-QN-CNN use an estimate over a batch instead of running estimate
                a, self.a_avg[m] = self.CovAHandler(input[0].data, m, bfgs=True)
                if not m in self.H_a:
                    batch_size, spatial_size = a.size(0), a.size(1)
                    a_ = a.view(-1, a.size(-1)) / spatial_size
                    cov_a = a_.t() @ (a_ / batch_size)
                    self.H_a[m] = torch.linalg.inv(cov_a + math.sqrt(self.damping) * torch.eye(cov_a.size(0)).to(cov_a.device))
                self.s_a[m] = self.H_a[m] @ self.a_avg[m].transpose(0, 1)
                s_a = self.s_a[m].view(self.s_a[m].size(0))
                batch_size, spatial_size = a.size(0), a.size(1)
                self.As[m] = torch.einsum('ntd,d->nt', (a, s_a)) # broadcasted dot product
                self.As[m] = torch.einsum('nt,ntd->ntd', (self.As[m], a)) # vector scaling
                self.As[m] = torch.einsum('ntd->d', self.As[m]) # sum over batch and spatial dim
                self.As[m] = self.As[m].unsqueeze(1) / batch_size
            elif m.__class__.__name__ == "Linear":
                aa, self.a_avg[m] = self.CovAHandler(input[0].data, m, bfgs=True)
                # initialize buffer
                if self.steps == 0:
                    self.m_aa[m] = torch.diag(aa.new(aa.size(0)).fill_(1))
                # KF-QN-FC use a running estimate
                update_running_stat(aa, self.m_aa[m], self.stat_decay)
                # initialize buffer
                if not m in self.H_a:
                    self.H_a[m] = torch.linalg.inv(self.m_aa[m] + math.sqrt(self.damping) * torch.eye(self.m_aa[m].size(0)).to(self.m_aa[m].device))

    def _save_pre_and_output(self, m, input, output):
        if self.steps % self.TCov == 0:
            self.pre_activations[m] = self.CovGHandler(output.data, m, batch_averaged=False, bfgs=False, pre=True)

            # initialize buffer
            if not m in self.s_g:
                self.s_g[m] = torch.zeros(self.pre_activations[m].size(-1), 1).to(self.pre_activations[m].device)
                self.y_g[m] = torch.zeros(self.pre_activations[m].size(-1), 1).to(self.pre_activations[m].device)

    def _save_next_pre(self, m, input, output):
        if self.steps % self.TCov == 0:
            self.next_pre_activations[m] = self.CovGHandler(output.data, m, batch_averaged=False, bfgs=False, pre=True)

    def _save_grad_output(self, m, grad_input, grad_output):
        if self.steps % self.TCov == 0:
            self.g_avg[m] = self.CovGHandler(grad_output[0].data, m, self.batch_averaged, bfgs=True)

    def _save_next_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.TCov == 0:
            self.next_g_avg[module] = self.CovGHandler(grad_output[0].data, module, self.batch_averaged, bfgs=True)

    def _prepare_model(self, model, cloned=False, init_module=False):
        if not cloned:
            if init_module:
                print("=> We keep following layers in KBFGS(L). ")

            count = 0
            for module in model.modules():
                if module.__class__.__name__ in self.known_modules:
                    if init_module:
                        self.modules.append(module)
                        print('(%s): %s' % (count, module))
                    self.handles.append(module.register_forward_pre_hook(self._save_input))
                    self.handles.append(module.register_forward_hook(self._save_pre_and_output))
                    self.handles.append(module.register_backward_hook(self._save_grad_output))

                    count += 1
        else:
            # print("=> We keep following layers (in cloned model) in KBFGS(L).")
            for module in model.modules():
                if module.__class__.__name__ in self.known_modules:
                    module.register_forward_hook(self._save_next_pre)
                    module.register_backward_hook(self._save_next_grad_output)
                    # print(module)

    def _get_LBFGS_Hv(self, m, v):
        if m not in self.s: # init
            return v
        is_scalar = (len(v.size()) == 1)
        if is_scalar:
            v = v.unsqueeze(1)

        gamma = self.gamma
        # if gamma == -1:
        #     gamma = 1 / self.R_inv[m][-1][-1].item() / self.yTy[m][-1][-1].item()
        assert(gamma > 0)

        Hv = gamma * v + torch.mm(self.left_matrix[m], torch.mm(self.right_matrix[m], v))
        if is_scalar:
            Hv = Hv.squeeze(1)
        return Hv

    def _get_BFGS_update(self, H, s, y, g_k=None):
        s = s.view(s.size(0))
        y = y.view(y.size(0))
        g_k = g_k.view(g_k.size(0))
        rho_inv = torch.dot(s, y)

        if rho_inv <= 0:
            return H, 1
        elif rho_inv <= 10**(-4) * torch.dot(s, s) * math.sqrt(torch.dot(g_k, g_k).item()):
            return H, 2
     
        rho = 1 / rho_inv

        Hy = torch.mv(H, y)
        H_new = H.data +\
        (rho**2 * torch.dot(y, torch.mv(H, y)) + rho) * torch.ger(s, s) -\
        rho * (torch.ger(s, Hy) + torch.ger(Hy, s))

        if torch.max(torch.isinf(H_new)):
            return H, 4
        else:
            H = H_new

        return H, 0

    def _get_DpDlm_damping(self, m, s, y, mu1, mu2):
        s = s.view(s.size(0))
        y = y.view(y.size(0))
        Hy = self._get_LBFGS_Hv(m, y)
        Hy = Hy.view(Hy.size(0))
        sTy = torch.dot(s, y)
        yHy = torch.dot(y, Hy)
        if sTy < mu1 * yHy:
            theta1 = (1 - mu1) * yHy / (yHy - sTy)
        else:
            theta1 = 1
        # Powell's damping on H
        s_ = theta1 * s + (1 - theta1) * Hy
        # LM damping on inv(H)
        y_ = y + mu2 * s_
        return s_, y_

    def _get_Powell_damping(self, m, s, y, mu1, mu2):
        sTy = torch.dot(s, y)
        Hy = self._get_LBFGS_Hv(m, y)
        Hy = Hy.view(Hy.size(0))
        yHy = torch.dot(y, Hy)
        div = sTy.item() / yHy.item()

        if div > mu1:
            theta = 1
            # status = 0
        else:
            theta = ((1 - mu1) * yHy / (yHy - sTy)).item()
            s = theta * s + (1 - theta) * Hy
            # status = 1
        return s, y

    def _get_modified_damping(self, m, s, y, mu1, mu2):
        sTs = torch.dot(s, s)
        sTy = torch.dot(s, y)

        if sTy / sTs > mu2:
            pass
            # status = 0
        else:
            theta = (1 - mu2) * sTs / (sTs - sTy)
            y = theta * y + (1 - theta) * s
            # status = 1
        return s, y

    def _get_DD_damping(self, m, s, y, mu1, mu2):
        s = s.view(s.size(0))
        y = y.view(y.size(0))
        s, y = self._get_Powell_damping(m, s, y, mu1, mu2)
        s, y = self._get_modified_damping(m, s, y, mu1, mu2)
        return s, y

    def _append_s_y(self, m, s, y):
        # col vec to mat
        s = s.unsqueeze(1)
        y = y.unsqueeze(1)

        if len(self.g_avg[m]) == 0:
            gTg = 0
        else:
            # g_avg is a row vec
            gTg = torch.mm(self.g_avg[m], self.g_avg[m].t()).item()

        yTs = torch.mm(y.t(), s)
        sTs = torch.mm(s.t(), s)

        if (not torch.isinf(sTs)) and (yTs.item() > 10**(-4) * sTs.item() * math.sqrt(gTg)):
            if m in self.s and (len(self.s[m]) == self.num_s_y_pairs):
                self.R_inv[m] = self.R_inv[m][1:, 1:]
                self.yTy[m] = self.yTy[m][1:, 1:]
                self.s[m] = self.s[m][1:]
                self.y[m] = self.y[m][1:]
                self.D_diag[m] = self.D_diag[m][1:]

            if m not in self.s: # init
                self.s[m] = s.t()
                self.y[m] = y.t()
            else:
                self.s[m] = torch.cat([self.s[m], s.t()], dim=0) # append row
                self.y[m] = torch.cat([self.y[m], y.t()], dim=0) # append row

            if m not in self.yTy: # init
                self.yTy[m] = torch.mm(self.y[m], self.y[m].t()) # self.y is row vec, dot product
            else:
                yT_new_y = torch.mm(self.y[m], y)
                self.yTy[m] = torch.cat([self.yTy[m], yT_new_y[:-1]], dim=1) # append col
                self.yTy[m] = torch.cat([self.yTy[m], yT_new_y.t()], dim=0) # append row

            if len(self.s[m]) == 1:
                self.D_diag[m] = torch.mm(self.s[m], self.y[m].t())
                self.D_diag[m] = self.D_diag[m].squeeze(0)

                self.R_inv[m] = 1 / self.D_diag[m][-1]
                self.R_inv[m] = self.R_inv[m].unsqueeze(0).unsqueeze(1)
            else:
                sTy = torch.mm(self.s[m], y)

                self.D_diag[m] = torch.cat([self.D_diag[m], sTy[-1]], dim=0)
                
                B = 1 / sTy[-1][-1]
                B = B.unsqueeze(0)
                B = B.unsqueeze(1)

                self.R_inv[m] = torch.cat([torch.cat([self.R_inv[m], torch.zeros(1, self.R_inv[m].size(1)).to(self.R_inv[m].device)], dim=0),
                                           torch.cat([-B * torch.mm(self.R_inv[m], sTy[:-1]), B], dim=0)],
                                           dim=1)
            gamma = self.gamma

            R_inv_sT = torch.mm(self.R_inv[m], self.s[m])

            if (m not in self.right_matrix) or (len(self.right_matrix[m]) < 2 * self.num_s_y_pairs):
                self.left_matrix[m] = torch.cat([R_inv_sT.t(), gamma * self.y[m].t()], dim=1)
                self.right_matrix[m] = torch.cat([torch.mm(torch.diag(self.D_diag[m]) + gamma * self.yTy[m], R_inv_sT) - gamma * self.y[m],
                                                  -R_inv_sT],
                                                  dim=0)
            else:
                self.left_matrix[m][:,:self.num_s_y_pairs] = R_inv_sT.t()
                self.left_matrix[m][:,self.num_s_y_pairs:] = gamma * self.y[m].t()
                self.right_matrix[m][:self.num_s_y_pairs] = self.D_diag[m][:, None] * R_inv_sT + gamma * (torch.mm(self.yTy[m], R_inv_sT) - self.y[m])
                self.right_matrix[m][self.num_s_y_pairs:] = -R_inv_sT

    def _update_inv(self, m, damping, n):
        """
        :param m: The layer
        :param n: copy of this layer (used as key for next_g_avg only)
        :return: no returns.
        """
        self.s_g[m] = self.stat_decay * self.s_g[m] + (1 - self.stat_decay) * (self.next_pre_activations[n] - self.pre_activations[m]).transpose(0, 1)
        self.y_g[m] = self.stat_decay * self.y_g[m] + (1 - self.stat_decay) * (self.next_g_avg[n] - self.g_avg[m]).transpose(0, 1)     
        if m.__class__.__name__ == 'Conv2d':
            s_g_, y_g_ = self._get_DpDlm_damping(m, self.s_g[m], self.y_g[m], 0.2, math.sqrt(damping))
        elif m.__class__.__name__ == 'Linear':
            s_g_, y_g_ = self._get_DD_damping(m, self.s_g[m], self.y_g[m], 0.2, math.sqrt(damping))
        else:
            raise NotImplementedError

        self._append_s_y(m, s_g_, y_g_)

        if m.__class__.__name__ == 'Conv2d':
            self.y_a[m] = self.As[m] + math.sqrt(damping) * self.s_a[m]
        elif m.__class__.__name__ == 'Linear':
            self.s_a[m] = self.H_a[m] @ self.a_avg[m].transpose(0, 1)
            self.y_a[m] = self.m_aa[m] @ self.s_a[m] + math.sqrt(damping) * self.s_a[m]
        else:
            raise NotImplementedError

        self.H_a[m], status = self._get_BFGS_update(self.H_a[m], self.s_a[m], self.y_a[m], self.a_avg[m].transpose(0, 1))

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_layer_update_direction(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        v = self._get_LBFGS_Hv(m, p_grad_mat)
        v = torch.mm(v, self.H_a[m])

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0] and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]
        return v

    def _update_grad(self, modules, updates):
        l = 0
        for m in modules:
            v = updates[l]
            m.weight.grad.data.copy_(v[0])

            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
            l += 1

    def _step(self, closure):
        # TODO(bmu): complete for K-BFGS
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(1, d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']

        # compute update direction
        updates = {}
        # layer index, the key used to match the parameters in the model and clone model
        l = 0
        for m in self.modules: # Conv2D and Linear only
            p_grad_mat = self._get_matrix_form_grad(m, m.__class__.__name__)
            v = self._get_layer_update_direction(m, p_grad_mat, damping)
            updates[l] = v
            l += 1

        self._update_grad(self.modules, updates)
        self._step(closure)

        if self.steps % self.TInv == 0:
            # clone model and do another fw-bw pass over this batch to compute next h and Dh
            # in order to update Hg
            # print('=> Another fw-bw pass for the following layers in KBFGS.')

            for handle in self.handles:
                handle.remove()
            self.handles.clear()

            model_new = copy.deepcopy(self.model)

            self._prepare_model(model_new, cloned=True)

            inputs, targets, criterion = closure()

            next_outputs = model_new.forward(inputs)
            next_loss = criterion(next_outputs, targets)

            model_new.zero_grad()
            next_loss.backward()

            inputs.detach()
            targets.detach()
            next_outputs.detach()
            next_loss.detach()

            new_modules = []

            for module in model_new.modules():
                if module.__class__.__name__ in self.known_modules:
                    new_modules.append(module)
                    # print('%s' % module)

            l = 0 # layer index, the key used to match the parameters in the model and clone model
            for m in self.modules:
                classname = m.__class__.__name__
                n = new_modules[l]
                self._update_inv(m, damping, n)
                l += 1

            # clean up memory of intermediate values (not needed any more) to update the inverse
            self.pre_activations.clear()
            self.next_pre_activations.clear()
            self.next_g_avg.clear()
            new_modules.clear()
            del model_new
            torch.cuda.empty_cache()

            self._prepare_model(self.model, cloned=False, init_module=False)

        self.steps += 1
