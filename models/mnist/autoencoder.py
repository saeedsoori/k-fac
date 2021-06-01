import torch.nn as nn


__all__ = ['autoencoder']


class Autoencoder(nn.Module):

    def __init__(self, num_classes=10, **kwargs):
        
        super(Autoencoder, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 30),
            nn.Linear(30, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 784)
        )

    def forward(self, x, bfgs=False):
        x = self.features(x)
        return x


def autoencoder(**kwargs):
    model = Autoencoder(**kwargs)
    return model
