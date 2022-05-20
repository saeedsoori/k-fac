import torch
import torch.nn as nn
import torch.nn.functional as F
# from backpack.utils.conv import unfold_func


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def update_running_stat(aa, m_aa, stat_decay):
    # using inplace operation to save memory!
    # print('update_running_stat')
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)

    # print('m_aa:\n', m_aa)
    # print('stat_decay:\n', stat_decay)


class ComputeMatGrad:

    @classmethod
    def __call__(cls, input, grad_output, layer):
        if isinstance(layer, nn.Linear):
            grad = cls.linear(input, grad_output, layer)
        elif isinstance(layer, nn.Conv2d):
            grad = cls.conv2d(input, grad_output, layer)
        else:
            raise NotImplementedError
        return grad

    @staticmethod
    def linear(input, grad_output, layer):
        """
        :param input: batch_size * input_dim
        :param grad_output: batch_size * output_dim
        :param layer: [nn.module] output_dim * input_dim
        :return: batch_size * output_dim * (input_dim + [1 if with bias])
        """
        with torch.no_grad():
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.unsqueeze(1)
            grad_output = grad_output.unsqueeze(2)
            grad = torch.bmm(grad_output, input)
        return grad

    @staticmethod
    def conv2d(input, grad_output, layer):
        """
        :param input: batch_size * in_c * in_h * in_w
        :param grad_output: batch_size * out_c * h * w
        :param layer: nn.module batch_size * out_c * (in_c*k_h*k_w + [1 if with bias])
        :return:
        """
        with torch.no_grad():
            input = _extract_patches(input, layer.kernel_size, layer.stride, layer.padding)
            input = input.view(-1, input.size(-1))  # b * hw * in_c*kh*kw
            grad_output = grad_output.transpose(1, 2).transpose(2, 3)
            grad_output = try_contiguous(grad_output).view(grad_output.size(0), -1, grad_output.size(-1))
            # b * hw * out_c
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.view(grad_output.size(0), -1, input.size(-1))  # b * hw * in_c*kh*kw
            grad = torch.einsum('abm,abn->amn', (grad_output, input))
        return grad

# TODO(bmu): check avg over spatial dim
class ComputeCovA:

    @classmethod
    def compute_cov_a(cls, a, layer, bfgs=False):
        return cls.__call__(a, layer, bfgs)

    @classmethod
    def __call__(cls, a, layer, bfgs=False):
        if isinstance(layer, nn.Linear):
            cov_a, a_avg = cls.linear(a, layer, bfgs)
        elif isinstance(layer, nn.Conv2d):
            cov_a, a_avg = cls.conv2d(a, layer, bfgs)
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            cov_a, a_avg = None, None

        if bfgs:
            return cov_a, a_avg
        return cov_a

    @staticmethod
    def conv2d(a, layer, bfgs=False):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        a_patch = a.view(a.size(0), a.size(1) * a.size(2), -1)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

        # a averaged over batch + spatial dimension
        if bfgs:
            a_avg = torch.mean(a, dim=0, keepdim=True)
            if layer.bias is not None:
                a_patch = torch.cat([a_patch, a_patch.new(a_patch.size(0), a_patch.size(1), 1).fill_(1)], 2)
            return a_patch, a_avg

        # FIXME(CW): do we need to divide the output feature map's size?
        a = a / spatial_size
        return a.t() @ (a / batch_size), None

    @staticmethod
    def linear(a, layer, bfgs=False):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

        # a averaged over batch dimension
        if bfgs:
            a_avg = torch.mean(a, dim=0, keepdim=True)
            return a.t() @ (a / batch_size), a_avg

        return a.t() @ (a / batch_size), None

class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False, bfgs=False, pre=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged, bfgs, pre)

    @classmethod
    def __call__(cls, g, layer, batch_averaged, bfgs=False, pre=False):
        if isinstance(layer, nn.Conv2d):
            cov_g, g_avg = cls.conv2d(g, layer, batch_averaged, bfgs, pre)
        elif isinstance(layer, nn.Linear):
            cov_g, g_avg  = cls.linear(g, layer, batch_averaged, bfgs, pre)
        else:
            cov_g, g_avg = None, None

        if pre or bfgs:
            return g_avg

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged, bfgs=False, pre=False):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]

        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size

        if pre or bfgs:
            avg_g = torch.mean(g, dim=0, keepdim=True)
            return None, avg_g

        g = g * spatial_size

        cov_g = g.t() @ (g / g.size(0))

        return cov_g, None

    @staticmethod
    def linear(g, layer, batch_averaged, bfgs=False, pre=False):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if pre or bfgs:
            if batch_averaged:
                g = g * batch_size
            avg_g = torch.mean(g, dim=0, keepdim=True)
            return None, avg_g

        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)

        return cov_g, None

if __name__ == '__main__':
    def test_ComputeCovA():
        pass

    def test_ComputeCovG():
        pass






