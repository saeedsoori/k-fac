import torch
import torch.nn as nn
import torch.nn.functional as F


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


def update_running_stat(cur, buf, stat_decay):
    # using inplace operation to save memory!
    buf *= (1 - stat_decay) / stat_decay
    buf += cur
    buf *= stat_decay


class ComputeCovA:

    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            cov_a, a_avg = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            cov_a, a_avg = cls.conv2d(a, layer)
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            cov_a, a_avg = None, None

        return cov_a, a_avg

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding) # [n, sh, sw, c_in * kh * kw]
        a = a.view(a.size(0), a.size(1) * a.size(2), -1) # [n, sh * sw, c_in * kh * kw]

        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), a.size(1), 1).fill_(1)], 2)

        a_avg = torch.mean(a, dim=1, keepdim=False) # [n, c_in * kh * kw]

        # FIXME(CW): do we need to divide the output feature map's size?
        return a_avg @ (a_avg.t() / batch_size), a_avg

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

        return a @ (a.t() / batch_size), a


class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            cov_g, g_avg = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g, g_avg = cls.linear(g, layer, batch_averaged)
        else:
            cov_g, g_avg = None, None

        return cov_g, g_avg

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        batch_size = g.shape[0]

        g = g.view(g.size(0), g.size(1), g.size(2) * g.size(3)) # [n, c_out, sh * sw]
        g_avg = torch.mean(g, dim=2, keepdim=False) # [n, c_out]

        if batch_averaged:
            g_avg = g_avg * batch_size

        cov_g = g_avg @ (g_avg.t() / batch_size)

        return cov_g, g_avg 

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if batch_averaged:
            cov_g = g @ (g.t() * batch_size)
        else:
            cov_g = g @ (g.t() / batch_size)
        return cov_g, g
