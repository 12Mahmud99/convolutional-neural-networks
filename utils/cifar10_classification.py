import torch 

def train(model, dataloader_train,dataload_val, le=1e-3, epochs=100, batch_size=64, patience=100):
    model.train()
    count = 0
