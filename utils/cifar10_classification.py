import torch
import torch.nn as nn

def train(model, dataloader_train,dataload_val, le=1e-3, epochs=100, batch_size=64, patience=100):
    model.train()
    count = 0
    optimizer = torch.optim.SGD() #minibatch
    for image, label in enumerate(dataloader_train):
        optimizer.zero_grads() #zero out the gradients
        output = model(image)

        loss = torch.nn.functional.CrossEntropy(output, label)
        loss.backward()# computer gradients
        optimizer.step()

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=4),
            nn.ReLU(), 
            nn.MaxPool2d((3,3), stride=2),
            nn.Linear(4096,1000),
            nn.Linear(1000,10))
        
    def forward(self,x):
        return self.layers(x)        