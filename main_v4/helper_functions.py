# The helper functions
# Author: Yizhao Shi, Xiangyu Han
# Date: 2023.12
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import torch.nn.functional as F
from yoloface import face_analysis
import cv2

cudnn.benchmark = True
plt.ion()   # interactive mode

face_detector = face_analysis()

################ Train the model #################
def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


################ Predict the results #################
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def predict(img, model, class_names):
    # Convert the numpy array (OpenCV image) to a PIL image
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply the transformation to the image
    input_tensor = transform(img)

    # Add a batch dimension
    input_tensor = input_tensor.unsqueeze(0)  # Adds a batch dimension at the beginning

    # Move tensor to GPU if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Set the model to evaluation mode
    model.eval()

    # No need to track gradients for prediction
    with torch.no_grad():
        # Make the prediction
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        max_prob, predicted_label = torch.max(probabilities, dim=1)

        # Get the corresponding class name
        predicted_class = class_names[predicted_label.item()]

        return max_prob.item(), predicted_class


def detect_face(img):
    # Detect faces and get face bounding boxes
    _, face_boxes, _ = face_detector.face_detection(img, model='tiny')

    # Initialize a list to store cropped faces and their coordinates
    cropped_faces = []
    face_coordinates = []

    # Extract the cropped area and save the face and its coordinates
    for i, box in enumerate(face_boxes):
        x, y, w, h = box
        cropped_face = img[y:y+w, x:x+h]  # Extract the cropped area
        cropped_faces.append(cropped_face)
        face_coordinates.append((x, y, w, h))

    # Return the list of cropped faces and their coordinates
    return cropped_faces, face_coordinates

    

# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title(f'predicted: {class_names[preds[j]]}')
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)