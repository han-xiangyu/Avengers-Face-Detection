# Train face recognition model
# Author: Yizhao Shi, Xiangyu Han
# Date: 2023.12
from facenet_pytorch import InceptionResnetV1
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
            transforms.Resize((160, 160)),  # Resize to 224x224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize((160, 160)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = './dataset/whole_dataset6'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=64,shuffle=True)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ########################### Load the pretrained model #################################
    myResnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',  # or use 'casia-webface' as required
        num_classes=len(class_names)
    ).to(device)

    # Freeze all layers initially
    for param in myResnet.parameters():
        param.requires_grad = False

    # Unfreeze the deeper layers for fine-tuning
    for param in myResnet.block8.parameters():
        param.requires_grad = True

    for param in myResnet.last_linear.parameters():
        param.requires_grad = True

    if myResnet.classify and hasattr(myResnet, 'logits'):
        for param in myResnet.logits.parameters():
            param.requires_grad = True


    # Modifying the Classifier
    # num_ftrs = myResnet.classifier[6].in_features
    # myResnet.classifier[6] = nn.Linear(num_ftrs, len(class_names))

    myResnet = myResnet.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(myResnet.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    ########################### Train and save model (fine tuning) #################################
    myResnet = train_model(dataloaders, dataset_sizes, myResnet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
    # # Save the model
    torch.save(myResnet.state_dict(), './models/model_avengers_ResNet_V3_fine_tune.pth')

if __name__ == '__main__':
    main()