
import torch.nn as nn


__all__ = ['fc']


class FC(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        super(FC, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(28*28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
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
        

    def forward(self, x):
        x = self.features(x)
        return x


def fc(**kwargs):
    """fully connected model for mnist.
    """
    model = FC(**kwargs)
    return model
