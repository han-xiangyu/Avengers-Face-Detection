# Train myVGG
# Author: Yizhao Shi, Xiangyu Han
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

    data_dir = './dataset/whole_dataset5'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=64,shuffle=True)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes


    # data_dir = 'data/hymenoptera_data'
    # # Load the entire dataset
    # full_dataset = datasets.ImageFolder(os.path.join(data_dir))

    # # Split the dataset into training and validation sets
    # train_size = int(0.8 * len(full_dataset))  # 80% of data for training
    # val_size = len(full_dataset) - train_size  # Remaining 20% for validation
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # # Transform the datasets
    # train_dataset = data_transforms(train_dataset,'train')
    # val_dataset = data_transforms(val_dataset,'val')

    # # Create DataLoaders for Training and Validation
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=128)
    # val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=128)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ########################### Load the pretrained model #################################
    myVGG16 = models.vgg16(pretrained=True)

    # Freeze all layers initially
    for param in myVGG16.parameters():
        param.requires_grad = False

    # Unfreeze the deeper layers for fine-tuning
    for param in myVGG16.features[-2:].parameters():
        param.requires_grad = True
    for param in myVGG16.classifier.parameters():
        param.requires_grad = True

    # Modifying the Classifier
    num_ftrs = myVGG16.classifier[6].in_features
    myVGG16.classifier[6] = nn.Linear(num_ftrs, len(class_names))

    myVGG16 = myVGG16.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(myVGG16.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    ########################### Train and save model (fine tuning) #################################
    myVGG16 = train_model(dataloaders, dataset_sizes, myVGG16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=13)
    # # Save the model
    torch.save(myVGG16.state_dict(), './models/model_avengers_V5.1.pth')

if __name__ == '__main__':
    main()