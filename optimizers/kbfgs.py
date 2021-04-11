import math
import copy

import torch
import torch.optim as optim

from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat


class KBFGSOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 weight_decay=0,
                 stat_decay=0.9,
                 damping=0.3,
                 TCov=1,
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
        # TODO (CW): optimizer now only support model as input
        super(KBFGSOptimizer, self).__init__(model.parameters(), defaults)
        self.handles = []
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.pre_activations, self.next_pre_activations = {}, {}

        self.model = model
        self._prepare_model(self.model, init_module=True)

        self.steps = 0

        self.stat_decay = stat_decay
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.damping = damping
        self.lr = lr

        self.TCov = TCov
        self.TInv = TInv

        # BFGS specific
        # TODO(bmu): may compute As_a without explicitly computing A
        self.m_aa = {}

        # a, g averaged over batch + spatial dimension for conv; over batch for fc
        self.a_avg, self.g_avg, self.next_g_avg = {}, {}, {}

        self.s_a, self.y_a, self.H_a = {}, {}, {}
        self.s_g, self.y_g, self.H_g = {}, {}, {}


    def _save_input(self, m, input):
        if torch.is_grad_enabled(): # TODO(bmu): clean?
            # KF-QN-CNN use an estimate over a batch instead of running estimate
            self.m_aa[m], self.a_avg[m] = self.CovAHandler(input[0].data, m, bfgs=True)

            # initialize
            # if self.steps == 0:
            if not m in self.H_a:
                self.H_a[m] = torch.linalg.inv(self.m_aa[m] + math.sqrt(self.damping) * torch.eye(self.m_aa[m].size(0)))

    def _save_pre_and_output(self, m, input, output):
        self.pre_activations[m] = self.CovGHandler(output, m, batch_averaged=False, bfgs=False, pre=True)

        # TODO(bmu): enable .grad for non-leaf tensor?
        # output.retain_grad()
        # TODO(bmu): initialize buffers
        # if self.steps == 0:
        if not m in self.H_g:
            self.H_g[m] = torch.eye(self.pre_activations[m].size(-1))
            self.s_g[m] = torch.zeros(self.pre_activations[m].size(-1), 1)
            self.y_g[m] = torch.zeros(self.pre_activations[m].size(-1), 1)

    def _save_next_pre(self, m, input, output):
        self.next_pre_activations[m] = self.CovGHandler(output, m, batch_averaged=False, bfgs=False, pre=True)

            # TODO(bmu): enable .grad for non-leaf tensor?
            # output.retain_grad()

    def _save_grad_output(self, m, grad_input, grad_output):
        self.g_avg[m] = self.CovGHandler(grad_output[0].data, m, self.batch_averaged, bfgs=True)

    def _save_next_grad_output(self, module, grad_input, grad_output):
        self.next_g_avg[module] = self.CovGHandler(grad_output[0].data, module, self.batch_averaged, bfgs=True)

    def _prepare_model(self, model, cloned=False, init_module=False):
        print(model)

        if not cloned:
            count = 0
            print("=> We keep following layers in KBFGS. ")
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
            print("=> We keep following layers (in cloned model) in KBFGS.")
            for module in model.modules():
                if module.__class__.__name__ in self.known_modules:
                    module.register_forward_hook(self._save_next_pre)
                    module.register_backward_hook(self._save_next_grad_output)
                    print(module)

    def _get_BFGS_update(self, H, s, y, g_k=None):
        # TODO(bmu): check if inplace operation
        s = s.view(s.size(0))
        y = y.view(y.size(0))
        rho_inv = torch.dot(s, y)

        if rho_inv <= 0:
            return H, 1
        # TODO(bmu): complete case 2 with g_k
        # elif rho_inv <= 10**(-4) * torch.dot(s, s) * math.sqrt(torch.dot(g_k, g_k).item()):
        #     return H, 2
     
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

    def _get_DD_damping(self, H, s, y, mu1, mu2):
        s = s.view(s.size(0))
        y = y.view(y.size(0))
        v0 = torch.mv(H, y)
        v1 = torch.dot(s, y)
        v2 = torch.dot(y, v0)
        if v1 < mu1 * v2:
            theta1 = (1 - mu1) * v2 / (v2 - v1)
        else:
            theta1 = 1
        # Powell's damping on H
        s_ = theta1 * s + (1 - theta1) * v0
        # LM damping on inv(H)
        y_ = y + mu2 * s_
        return s_, y_

    def _update_inv(self, m, damping, n):
        """
        :param m: The layer
        :param n: copy of this layer (used as key for next_g_avg only)
        :return: no returns.
        """
        self.s_a[m] = self.H_a[m] @ self.a_avg[m].transpose(0, 1)
        self.y_a[m] = self.m_aa[m] @ self.s_a[m] + math.sqrt(damping) * self.s_a[m]
        self.H_a[m], status = self._get_BFGS_update(self.H_a[m], self.s_a[m], self.y_a[m])

        self.s_g[m] = self.stat_decay * self.s_g[m] + (1 - self.stat_decay) * (self.next_pre_activations[n] - self.pre_activations[m]).transpose(0, 1)
        self.y_g[m] = self.stat_decay * self.y_g[m] + (1 - self.stat_decay) * (self.next_g_avg[n] - self.g_avg[m]).transpose(0, 1)

        s_g_, y_g_ = self._get_DD_damping(self.H_g[m], self.s_g[m], self.y_g[m], 0.2, math.sqrt(damping))

        self.H_g[m], status = self._get_BFGS_update(self.H_g[m], s_g_, y_g_)

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
        # p_grad_mat is of output_dim * input_dim
        v = self.H_g[m] @ p_grad_mat @ self.H_a[m]

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0] and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _update_grad(self, modules, updates, step=False):
        l = 0
        for m in modules:
            v = updates[l]
            m.weight.grad.data.copy_(v[0])
            if step:
                # if self.weight_decay != 0 and self.steps >= 20 * self.TCov:
                # m.weight.grad.data.add_(self.weight_decay, m.weight.data)
                if self.momentum != 0:
                    param_state = self.state[m.weight]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(m.weight.data)
                        buf.mul_(self.momentum).add_(1, m.weight.grad.data)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(self.momentum).add_(1, m.weight.grad.data)
                    m.weight.grad.data = buf
                m.weight.data.add_(-self.lr, m.weight.grad.data)

            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                if step:
                    # if self.weight_decay != 0 and self.steps >= 20 * self.TCov:
                    # m.bias.grad.data.add_(self.weight_decay, m.bias.data)
                    if self.momentum != 0:
                        param_state = self.state[m.bias]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(m.bias.data)
                            buf.mul_(self.momentum).add_(1, m.bias.grad.data)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(self.momentum).add_(1, m.bias.grad.data)
                        m.bias.grad.data = buf
                    m.bias.data.add_(-self.lr, m.bias.grad.data)
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
                # if weight_decay != 0 and self.steps >= 20 * self.TCov:
                # d_p.add_(weight_decay, p.data)
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

        # self._prepare_model(self.model, cloned=False)
        # self.steps += 1

        # return 

        # clone model and do another fw-bw pass over this batch to compute next h and Dh
        # in order to update Hg
        print('=> Another fw-bw pass for the following layers in KBFGS.')

        for handle in self.handles:
            handle.remove()
            # pass      

        model_new = copy.deepcopy(self.model)

        self._prepare_model(model_new, cloned=True)

        inputs, targets, criterion = closure()

        next_outputs = model_new.forward(inputs)
        next_loss = criterion(next_outputs, targets)

        model_new.zero_grad()
        next_loss.backward()

        new_modules = []

        for module in model_new.modules():
            if module.__class__.__name__ in self.known_modules:
                new_modules.append(module)
                print('%s' % module)


        # TODO(bmu): complete inverse update with BFGS below
        l = 0 # layer index, the key used to match the parameters in the model and clone model
        for m in self.modules:
            classname = m.__class__.__name__
            # if self.steps % self.TInv == 0:
            n = new_modules[l]
            self._update_inv(m, damping, n)
            l += 1

        
