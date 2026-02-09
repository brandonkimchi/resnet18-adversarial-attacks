#Importing libraries
import numpy as np
from torch import nn
import torch
from torch import optim 
import torchvision
from torchvision import models
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

# Transform train and validation set
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Get data and split into train and validation set
full_dataset = ImageFolder(
    r"C:/Users/lizzy/Documents/Deep Learning/archive/data/train",
    transform=train_transform
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size]
)

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# CNN neural network: ResNet-18
res_18_model = models.resnet18(pretrained=True)
res_18_model.fc= nn.Linear(512, 10)

model=res_18_model
if(torch.cuda.is_available()==True):
    model=res_18_model.cuda()

# Optimization
optimiser=optim.SGD(model.parameters(),lr=1e-2)
loss=nn.CrossEntropyLoss()

# My training and validation loops
nb_epochs = 4
acc_tot=np.zeros(nb_epochs)
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    model.train()     
    for x,y in train_loader: 

        if(torch.cuda.is_available()==True):
            x=x.cuda()
            y=y.cuda()        

        # 1 forward
        l = model(x)

        #2 compute the cost function
        J = loss(l,y)

        # 3 cleaning the gradients
        model.zero_grad()
        # optimiser.zero_grad()
        # params.grad.zero_()

        # 4 accumulate the partial derivatives of J wrt params
        J.backward()

        # 5 step in the opposite direction of the gradient
        optimiser.step()

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}', end=', ')
    print(f'training loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}')
    
    losses = list()
    accuracies = list() 
    model.eval()
    for batch in val_loader: 
        x,y = batch
        if(torch.cuda.is_available()==True):
            x=x.cuda()
            y=y.cuda()
    
        with torch.no_grad(): 
            l = model(x)
    
        #2 compute the objective function
        J = loss(l,y)
    
        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
    
    print(f'Epoch {epoch + 1}',end=', ')
    print(f'validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'validation accuracy: {torch.tensor(accuracies).mean():.2f}')
    acc_tot[epoch]=torch.tensor(accuracies).mean().numpy()

#torch.save(model.state_dict(), "resnet18_animal2.0.pth")











