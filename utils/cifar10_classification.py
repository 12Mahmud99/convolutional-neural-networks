import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train(model, dataloader_train, dataloader_val=None, epochs=1):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for batch_index, (image, label) in enumerate(dataloader_train):
            optimizer.zero_grad() 
            output = model(image) 

            loss = torch.nn.functional.cross_entropy(output, label) 
            loss.backward()
            optimizer.step()

def test(model, dataset_test):
    correct_predictions = 0
    total_samples = dataset_test.dataset.__len__()
    correct_label_to_prediction_count = {}
    model.eval() 
    for batch_index, (image, label) in enumerate(dataset_test):
        with torch.no_grad():
            output = model(image)

            correct_predictions += (output.argmax(dim=1) == label).sum().item()
    
    return correct_predictions / total_samples
