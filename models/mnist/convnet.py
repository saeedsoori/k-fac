
import torch.nn as nn


__all__ = ['convnet']


class ConvNet(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        # super(ConvNet, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(), 
        #     nn.Linear(28*28*32, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 10)         
        # )
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(28*28*1, 10),
        
        )

    def forward(self, x, bfgs=False):
        if bfgs:
            layer_inputs = []
            pre_activations = []

            # Assume Conv2d is always followed by activation
            # and Linear always have a Flatten before it
            assert(len(self.modules()) % 2 == 0)
            num_layer_groups = len(self.modules()) / 2
            for l in range(num_layer_groups):
                module = self.modules()[l * 2]
                act = self.modules()[l * 2 + 1]
                module_name = module.__class__.__name__
                act_name = act.__class__.__name__
                print('* forward [' + module_name + '] + [' + act_name + ']')
                layer_inputs.append(x)
                pre = module(x)
                pre_activations.append(pre)
                x = act(x)
                pre.retain_grad() # enable .grad for non-leaf tensor
            return x, layer_inputs, pre_activations

        x = self.features(x)
        return x


def convnet(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = ConvNet(**kwargs)
    return model
