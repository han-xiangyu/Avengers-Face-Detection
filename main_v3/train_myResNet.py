# Train myVGG
# Author: Xiangyu Han
# Date: 2023.12
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from helper_functions import *
from torch.utils.data import random_split

def main():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = './dataset/whole_dataset6_2'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=64,shuffle=True)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the pretrained ResNet model
    myResNet = models.resnet50(pretrained=True)  # You can choose other versions of ResNet as needed

    # Freeze all layers
    for param in myResNet.parameters():
        param.requires_grad = False

    # # Modify the classifier to fit your task
    # num_ftrs = myResNet.fc.in_features
    # myResNet.fc = nn.Linear(num_ftrs, len(class_names))  # Replace the last fully connected layer

    # Unfreeze the last convolution layer and the fully connected layer
    for param in myResNet.layer4.parameters():
        param.requires_grad = True

    myResNet.fc = nn.Linear(myResNet.fc.in_features, len(class_names))  # Replace the last fully connected layer

    # Move the model to the selected device
    myResNet = myResNet.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, myResNet.parameters()), lr=0.001)

    # Define the learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # Train the model (the train_model function needs to be modified according to your implementation)
    myResNet = train_model(dataloaders, dataset_sizes, myResNet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=13)

    # Save the model
    torch.save(myResNet.state_dict(), './models/model_avengers_ResNet_6_2.pth')

if __name__ == '__main__':
    main()
