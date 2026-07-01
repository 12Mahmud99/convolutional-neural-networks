import torch 
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import cifar10

#alexne has 8 learnable layers as follows
# ReLU(Conv2D) -> MaxPool -> ReLU(Conv2D) -> MaxPool -> ReLU(Conv2D) -> ReLU(Conv2D) -> ReLU(Conv2D) -> MaxPool
# -> linear -> linear -> linear
class AlexNet(nn.Module):
    def __init__():
        self.layers=nn.Sequential(nn.Conv2D(batch_size=64, in_channels=3, out_channels=96, 
            kernel_size=(11,11), stride=4),
            nn.ReLU(), 
            nn.MaxPool((3,3), stride=2),
            nn.Conv2D(batch_size=64, in_channels=3, out_channels=256, 
            kernel_size=(5,5), stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool((3,3), stride=2),
            nn.Conv2D(batch_size=64, in_channels=3, out_channels=384, 
            kernel_size=(3,3), stride=4),
            nn.ReLU(),
            nn.Conv2D(batch_size=64, in_channels=3, out_channels=256, 
            kernel_size=(3,3), stride=4),
            nn.ReLU(),
            nn.MaxPool((3,3), stride=2),
            nn.Linear(4096,4096),
            nn.Linear(4096,1000)
            nn.Linear(1000,10))