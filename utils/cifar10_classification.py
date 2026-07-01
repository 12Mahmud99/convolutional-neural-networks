import torch

def train(model, dataloader_train,dataload_val, le=1e-3, epochs=100, batch_size=64, patience=100):
    model.train()
    count = 0
    optimizer = torch.optim.SGD() #minibatch
    for image, label in enumerate(dataloader_train):
        optimizer.zero_grads() #zero out the gradients