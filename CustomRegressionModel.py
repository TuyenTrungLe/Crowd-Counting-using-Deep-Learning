import torch.nn as nn

class CustomRegressionModel(nn.Module):
    def __init__(self, feature_extractor):
        super(CustomRegressionModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(60 * 80, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x