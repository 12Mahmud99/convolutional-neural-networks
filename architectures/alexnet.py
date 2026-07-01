import torch 
from torch import nn
import torchvision

#alexne has 8 learnable layers as follows
# ReLU(Conv2D) -> MaxPool -> ReLU(Conv2D) -> MaxPool -> ReLU(Conv2D) -> ReLU(Conv2D) -> ReLU(Conv2D) -> MaxPool
# -> linear -> linear
class AlexNet(nn.Module):
    def __init__():
        self.layers=nn.Sequential(nn.Conv2D(batch_size=64, in_channels=3, out_channels=96, 
            kernel_size=(11,11), stride=4), 
            nn.MaxPool(),)