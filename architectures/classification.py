from torch import nn

#BasicNet
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

##AlexNet (2012) #expected images of size 227x227x3
class AlexNet227(nn.Module):
    def __init__(self):
        super().__init__()        
        self.layers=nn.Sequential(
            #first conv layer
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11),stride=4, padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3,3),stride=2),
            
            #second conv layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            
            #third conv layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            
            #fourth conv layer
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            
            #fifth conv layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=1, padding=1),                   
            nn.ReLU(),
            
            #local reponse normalization mimics lateral inhibition in neuroscience
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3,3),stride=2),
            
            #first classifier layer
            nn.Flatten(),
            nn.Linear(6*6*256,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            #second classifier layer
            nn.Linear(4096,4096),
            nn.ReLU(), #cuz the weights could make it negative again
            nn.Dropout(p=0.5),
                       
            #thirds classifier layer
            nn.Linear(4096,1000), #output size is teh number of classes        
        )
        
        ##the loss function is always in the training loop outside the architectures 
    def forward(self, x):
        return self.layers(x)


##AlexNet32 ##AlexNet for 32x32 images
class AlexNet32(nn.Module):
    def __init__(self):
        super().__init__()        
        self.layers=nn.Sequential(
            #first conv layer
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(2,2),stride=1, padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3,3),stride=2),
            
            #second conv layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(2,2), stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            
            #third conv layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(2,2), stride=1, padding=1),
            nn.ReLU(),
            
            #fourth conv layer
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1,1), stride=1, padding=1),
            nn.ReLU(),
            
            #fifth conv layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(1,1), stride=1, padding=1),                   
            nn.ReLU(),

            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3,3),stride=2),
            
            #first classifier layer
            nn.Flatten(),
            nn.Linear(6*6*256,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            #second classifier layer
            nn.Linear(4096,4096),
            nn.ReLU(), #cuz the weights could make it negative again
            nn.Dropout(p=0.5),
                       
            #thirds classifier layer
            nn.Linear(4096,10), #output size is teh number of classes        
        )
        
        ##the loss function is always in the training loop outside the architectures 
    def forward(self, x):
        return self.layers(x)

##VGG-16 
class VGG16(nn.Module):
    def __init__(self):
        super.__init__()
        #TODO
        
    def forward(self, x):
        ##TODO
        pass

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return nn.ReLU(self.conv_path(x) + x)

##BasicResNet
class BasicResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.res_block1 = ResidualBlock(32)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.init_conv(x)   
        x = self.res_block1(x)  
        x = self.gap(x)         # Shape: [batch, 32, 1, 1]
        x = nn.flatten(x, 1) # Shape: [batch, 32]
        return self.fc(x)       # Shape: [batch, 10]

##ResNet-50
class ResNet50(nn.Module):
    def __init__(self):
        super.__init__()
        #TODO
        
    def forward(self, x):
        ##TODO
        pass