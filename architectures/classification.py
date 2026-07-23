from torch import nn

class BasicNet(nn.Module):
    def __init__(self, pooling="max"):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=1),
            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(46464, 10), 
        )
        
    def forward(self, x):
        return self.layers(x)

##AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_channel=3, out_channels=96)
        )
    def forward(self, x):
        

##VGG-16 
class VGG16(nn.Module):
    def __init__(self):
        super.__init__()
        #TODO
        
    def forward(self, x):
        ##TODO
        pass

##ResNet-50
class ResNet50(nn.Module):
    def __init__(self):
        super.__init__()
        #TODO
        
    def forward(self, x):
        ##TODO
        pass