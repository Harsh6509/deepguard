import torch
import torch.nn as nn
from models.spatial_model import SpatialCNN

class HybridDetector(nn.Module):

    def __init__(self):

        super().__init__()

        self.spatial = SpatialCNN()

        self.fc = nn.Sequential(

            nn.Linear(128*16*16,128),
            nn.ReLU(),

            nn.Linear(128,3)
        )

    def forward(self,x):

        x = self.spatial(x)
        x = self.fc(x)

        return x