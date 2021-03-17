
import torch.nn as nn


__all__ = ['fc']


class FC(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        super(FC, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(28*28, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )
        # self.features = nn.Sequential(
        #     nn.Flatten(), 
        #     nn.Linear(28*28, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 10)
        # )

    def forward(self, x):
        x = self.features(x)
        return x


def fc(**kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = FC(**kwargs)
    return model
