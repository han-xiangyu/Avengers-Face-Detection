# CIS5810 Final Project
# Author: Yizhao Shi, Xiangyu Han 
# Date: 2023.11
# Description: This file is used to train the model used for classification.
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from training_classifier import train_network
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from helper_functions import *

################## Define the model ##################
class myNetwork(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize1, hiddenSize2, dropout_rate=0.5):
        super(myNetwork, self).__init__()
        # First linear layer
        self.linear1 = nn.Linear(inputSize, hiddenSize1)
        # Dropout layer after the first linear layer
        self.dropout1 = nn.Dropout(dropout_rate)
        # Second linear layer
        self.linear2 = nn.Linear(hiddenSize1, hiddenSize2)
        # Dropout layer after the second linear layer
        self.dropout2 = nn.Dropout(dropout_rate)
        # Third linear layer that outputs the class scores
        self.linear3 = nn.Linear(hiddenSize2, outputSize)

    def forward(self, x):
        # Pass data through linear1, apply ReLU activation, then dropout
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        # Pass data through linear2, apply ReLU activation, then dropout
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        # Pass data through linear3 (no ReLU or dropout after this since it's the output layer)
        out = self.linear3(x)
        return out


if __name__ == "__main__":

    ################## Load the data for classification ##################
    with open('./dataset/feature_vectors_by_label_avengers.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    print(len(data_dict))

    labels = []
    feature_vectors = []
    for label, features in data_dict.items():
        for feature in features:
            labels.append(label)
            feature_vectors.append(feature)

    print(len(labels))
    print(len(feature_vectors))


    class MyDataset(Dataset):
        def __init__(self, data_dict):
            self.data = []  # data is the set of feature vectors
            self.labels = []
            for label, features in data_dict.items():
                for feature in features:
                    self.data.append(feature)
                    self.labels.append(label)


        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

    # Load Dataset
    dataset = MyDataset(data_dict)
    # Split Dataset into Training and Validation Sets
    train_size = int(0.8 * len(dataset))  # 80% of data for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create DataLoaders for Training and Validation
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    ################## Train the model ##################
    EPOCH = 50
    inputSize = 2622
    outputSize = 5
    hiddenSize1 = 128
    hiddenSize2 = 64

    # Define the model
    model = myNetwork(inputSize, outputSize, hiddenSize1, hiddenSize2)
    trained_model, preds, gt = train_network(model,train_loader,val_loader)

    # Save the model
    torch.save(trained_model.state_dict(), './dataset/model_avengers.pth')
