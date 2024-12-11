import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super(FeatureExtractor, self).__init__()
        self.csrnet = pretrained_model

    def forward(self, x):
        x = self.csrnet(x)
        return x