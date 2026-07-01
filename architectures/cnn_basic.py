import torch 
from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        self.layers=nn.Sequential(nn.Conv2D(batch_size=64, in_channels=3, out_channels=96, 
            kernel_size=(11,11), stride=4),
            nn.ReLU(), 
            nn.MaxPool((3,3), stride=2),
            nn.Linear(4096,1000),
            nn.Linear(1000,10))
        
    def forward(self,x):
        return self.layers(x)