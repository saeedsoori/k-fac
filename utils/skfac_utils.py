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
    def compute_cov_a(cls, a, layer, subsample='false', num_ss_patches=0):
        return cls.__call__(a, layer, subsample, num_ss_patches)

    @classmethod
    def __call__(cls, a, layer, subsample='false', num_ss_patches=0):
        if isinstance(layer, nn.Linear):
            cov_a, a_avg = cls.linear(a, layer, subsample, num_ss_patches)
        elif isinstance(layer, nn.Conv2d):
            cov_a, a_avg = cls.conv2d(a, layer, subsample, num_ss_patches)
        else:
            # FIXME(CW): for extension to other layers.
            # raise NotImplementedError
            cov_a, a_avg = None, None

        return cov_a, a_avg

    @staticmethod
    def conv2d(a, layer, subsample='false', num_ss_patches=0):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding) # [n, sh, sw, c_in * kh * kw]
        a = a.view(a.size(0), a.size(1) * a.size(2), -1) # [n, sh * sw, c_in * kh * kw]

        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), a.size(1), 1).fill_(1)], 2)

        if subsample == 'true':
            num_spatial_locations = a.size(1) # sh * sw
            sample_idx = torch.randint(low=0, high=num_spatial_locations, size=(num_ss_patches,))
            sample = a[:, sample_idx, :] # [n, num_ss_patches, c_in * kh * kw]
            a = sample.view(-1, a.size(-1)) # [n * num_ss_patches, c_in * kh * kw]
        else:
            a_avg = torch.sum(a, dim=1, keepdim=False) # [n, c_in * kh * kw]
            a = a_avg

        # FIXME(CW): do we need to divide the output feature map's size?
        return a @ (a.t() / batch_size), a

    @staticmethod
    def linear(a, layer, subsample='false', num_ss_patches=0):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

        return a @ (a.t() / batch_size), a


class ComputeCovG:

    @classmethod
    def compute_cov_g(cls, g, layer, subsample='false', num_ss_patches=0, batch_averaged=True):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, subsample, num_ss_patches, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, subsample='false', num_ss_patches=0, batch_averaged=True):
        if isinstance(layer, nn.Conv2d):
            cov_g, g_avg = cls.conv2d(g, layer, subsample, num_ss_patches, batch_averaged)
        elif isinstance(layer, nn.Linear):
            cov_g, g_avg = cls.linear(g, layer, subsample, num_ss_patches, batch_averaged)
        else:
            cov_g, g_avg = None, None

        return cov_g, g_avg

    @staticmethod
    def conv2d(g, layer, subsample='false', num_ss_patches=0, batch_averaged=True):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        batch_size = g.shape[0]

        g = g.view(g.size(0), g.size(1), g.size(2) * g.size(3)) # [n, c_out, sh * sw]

        if subsample == 'true':
            num_spatial_locations = g.size(-1) # sh * sw
            sample_idx = torch.randint(low=0, high=num_spatial_locations, size=(num_ss_patches,))
            sample = g[:, :, sample_idx]
            g = sample.transpose_(1, 2)
            g = g.reshape(-1, g.size(-1))
        else:
            g_avg = torch.sum(g, dim=2, keepdim=False) # [n, c_out]
            g = g_avg

        if batch_averaged:
            g = g * batch_size

        return g @ (g.t() / batch_size), g

    @staticmethod
    def linear(g, layer, subsample='false', num_ss_patches=0, batch_averaged=True):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if batch_averaged:
            cov_g = g @ (g.t() * batch_size)
        else:
            cov_g = g @ (g.t() / batch_size)
        return cov_g, g
