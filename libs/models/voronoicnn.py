__all__ = ['voronoi_cnn']


import torch.nn as nn


class VoronoiCNN(nn.Module):

    def __init__(self, in_chans):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_chans, 48, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, 7, padding=3)
        )
    
    def forward(self, x):
        return self.model(x)


def voronoi_cnn(**kwargs):
    model = VoronoiCNN(**kwargs)
    return model

