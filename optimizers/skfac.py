import math

import torch
import torch.optim as optim

from utils.skfac_utils import (ComputeCovA, ComputeCovG)
from utils.skfac_utils import update_running_stat


class SKFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): KFAC optimizer now only support model as input
        super(SKFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_a, self.m_g = {}, {} # moving average of A and DS (i.e. G)
        self.aaT, self.ggT = {}, {}
        self.a, self.g = {}, {}

        self.H_a, self.H_g = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            a = input[0].data

            # Initialize buffer
            if self.steps == 0:
                self.m_a[module] = torch.zeros_like(a)
            update_running_stat(a, self.m_a[module], self.stat_decay)
            self.aaT[module], self.a[module] = self.CovAHandler(self.m_a[module], module)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            g = grad_output[0].data

            # Initialize buffer
            if self.steps == 0:
                self.m_g[module] = torch.zeros_like(g)
            update_running_stat(g, self.m_g[module], self.stat_decay)
            self.ggT[module], self.g[module] = self.CovGHandler(self.m_g[module], module, self.batch_averaged)

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in SKFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in SKFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m, damping):
        """Do cholesky decomposition for computing inverse of the damped factors.
        :param m: The layer
        :return: no returns.
        """
        n = self.m_a[m].size(0)
        a, g = self.a[m], self.g[m]

        self.H_a[m] = a.t() @ torch.cholesky_inverse(self.aaT[m] + n * math.sqrt(damping) * torch.eye(n).to(self.aaT[m].device)) @ a
        self.H_a[m] = (torch.eye(self.H_a[m].size(0)).to(self.H_a[m].device) - self.H_a[m]) / math.sqrt(damping)

        self.H_g[m] = g.t() @ torch.cholesky_inverse(self.ggT[m] + n * math.sqrt(damping) * torch.eye(n).to(self.ggT[m].device)) @ g
        self.H_g[m] = (torch.eye(self.H_g[m].size(0)).to(self.H_g[m].device) - self.H_g[m]) / math.sqrt(damping)
        

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

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim

        v = self.H_g[m] @ p_grad_mat @ self.H_a[m]
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        # vg_sum = 0
        # for m in self.modules:
        #     v = updates[m]
        #     vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
        #     if m.bias is not None:
        #         vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        # nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            # m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                # m.bias.grad.data.mul_(nu)

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
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
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
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m, damping)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1