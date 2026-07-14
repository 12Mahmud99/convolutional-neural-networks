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
    model.eval() 
    for batch_index, (image, label) in enumerate(dataset_test):
        with torch.no_grad():
            output = model(image)

            num_images_to_show = min(10, image.size(0))
            for i in range(num_images_to_show):
                img_tensor = image[i].cpu()
                
                img_numpy = img_tensor.permute(1, 2, 0).numpy()
                
                plt.imshow(img_numpy)
                plt.axis('off')  
                plt.show()
                
                print(f"Predicted Class: {torch.argmax(output[i]).item()}")
