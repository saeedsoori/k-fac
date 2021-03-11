
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
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(28*28*8, 40),
            nn.ReLU(),
            nn.Linear(40, 100),
            nn.ReLU(),
            nn.Linear(100, 10)         
        )

    def forward(self, x):
        x = self.features(x)
        return x


def convnet(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = ConvNet(**kwargs)
    return model
