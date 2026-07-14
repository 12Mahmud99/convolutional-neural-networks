from torch import nn

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=1),
            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear(46464, 10), 
        )
        
    def forward(self, x):
        return self.layers(x)


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
    def forward():
        pass