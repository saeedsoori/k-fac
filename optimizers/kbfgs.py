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
                 stat_decay=0.9,
                 damping=0.3,
                 kl_clip=0.001,
                 weight_decay=0,
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
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv

        # BFGS specific
        # TODO(bmu): may compute As_a without explicitly computing A
        self.m_aa = {}

        # a, g averaged over batch + spatial dimension for conv; over batch for fc
        self.a_avg, self.g_avg = {}, {}

        self.s_a, self.y_a, self.H_a = {}, {}, {}
        self.s_g, self.y_g, self.H_g = {}, {}, {}


    def _save_input(self, module, input):
        if torch.is_grad_enabled():
            # KF-QN-CNN use an estimate over a batch instead of running estimate
            self.m_aa[module], self.a_avg[module] = self.CovAHandler(input[0].data, module, bfgs=True)

            # TODO(bmu): initialize buffers

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.acc_stats:
            self.g_avg[module] = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)

            # TODO(bmu): initialize buffers

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in KBFGS. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _get_BFGS_update(H, s, y, g_k=None):
        # TODO(bmu): check if inplace operation
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

    def _update_inv(self, m, damping):
        """
        :param m: The layer
        :return: no returns.
        """
        # TODO(bmu): keep?
        # eps = 1e-10  # for numerical stability

        if self.steps == 0:
            self.H_a[m] = torch.linalg.inv(self.m_aa[m] + math.sqrt(damping) * torch.eye(self.m_aa[m].size(0)))
        self.s_a[m] = self.H_a[m] @ self.a_avg[m]
        self.y_a[m] = self.m_aa[m] @ self.s_a[m] + math.sqrt(damping) * self.s_a[m]
        self.H_a[m], status = self._get_BFGS_update(self.H_a[m], self.s_a[m], self.y_a[m])
        print('BFGS update status: ', status)

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

    def _get_grad_update(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim

        print('=== _get_grad_update ===')
        print('self.H_g[m].shape: ', self.H_g[m].shape)
        print('p_grad_mat.shape: ', p_grad_mat.shape)
        print('self.H_a[m].shape: ', self.H_a[m].shape)

        v = self.H_g[m] @ p_grad_mat @ self.H_a[m]
        print('v.shape: ', v.shape)

        # TODO(bmu): check bias
        
        # v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        # v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        # v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        # if m.bias is not None:
        #     # we always put gradient w.r.t weight in [0]
        #     # and w.r.t bias in [1]
        #     v = [v[:, :-1], v[:, -1:]]
        #     v[0] = v[0].view(m.weight.grad.data.size())
        #     v[1] = v[1].view(m.bias.grad.data.size())
        # else:
        #     v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # TODO(bmu): remove?
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
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

    def step(self, closure=None):
        print("=== step ===")

        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']

        # get next grad
        model_new = copy.deepcopy(self.model)
        new_modules = []

        for module in model_new.modules():
            classname = module.__class__.__name__
            print('=> Another fw-bw pass for the following layers in KBFGS. <=')
            if classname in self.known_modules:
                new_modules.append(module)

        num_modules = len(self.modules)
        for l in range(num_modules): # known modules: Conv2d, Linear
            m = self.modules[l]
            classname = m.__class__.__name__
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_grad_update(m, p_grad_mat, damping)
            new_modules[l].weight.data += lr * v

        inputs, targets, criterion, outputs, layer_inputs, pre_activations = closure()
        next_outputs, next_layer_inputs, next_pre_activations = model_new.forward(inputs, bfgs=True)
        next_loss = criterion(next_outputs, targets)

        model_new.zero_grad()
        next_loss.backward()


        # TODO(bmu): complete inverse update with BFGS below
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m, damping)
            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_grad_update(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1
