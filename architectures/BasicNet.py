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
