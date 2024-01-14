import os
import cv2
import torch
import torchvision.models as models
from helper_functions import *
import sys

# Load the model
data_dir = './dataset/whole_dataset6_2'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Load the pre-trained ResNet model
myResNet = models.resnet50(pretrained=True)
num_ftrs = myResNet.fc.in_features

# Modify the last fully connected layer to fit your task
myResNet.fc = torch.nn.Linear(num_ftrs, len(class_names))

# Load the saved model weights (if previously trained)
myResNet.load_state_dict(torch.load('./models/model_avengers_ResNet_6_2.pth'))

# Set the model to evaluation mode
myResNet.eval()

if torch.cuda.is_available():
    myVGG16 = myResNet.cuda()

# Directory containing test images
test_image_directory = './dataset/whole_dataset6_2/test'

# Initialize variables for performance evaluation
total_predictions = 0
correct_predictions = 0

# output file name
output_file = './dataset/whole_dataset6_2/test_result.txt'

with open(output_file, 'a') as file:
  
    sys.stdout = file

    # Iterate over each subfolder (label) in the test directory
    for label in os.listdir(test_image_directory):
        label_path = os.path.join(test_image_directory, label)

        # Check if it's a directory
        if not os.path.isdir(label_path):
            continue

        # Iterate over each image in the subfolder
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            
            if not os.path.isfile(image_path):
                continue

            img = cv2.imread(image_path)
            print(image_path)
            # Use the detect_face function to get detected faces and their coordinates
            detected_faces, faces_coordinates = detect_face(img)

            # Skip evaluation if no faces are detected
            if detected_faces is None or len(detected_faces) == 0:
                continue

            # Loop over each detected face for prediction
            for i, face in enumerate(detected_faces):
                max_prob, predicted_label = predict(face, myVGG16,class_names)
                x, y, w, h = faces_coordinates[i]
                

                if max_prob < 0.7:
                    predicted_label = 'Unknown'

                print(predicted_label,max_prob)
                # Update performance metrics
                total_predictions += 1
                if predicted_label == label:
                    correct_predictions += 1

                # cv2.rectangle(img, (x, y), (x + h, y + w), (255, 0, 0), 2)
                # cv2.putText(img, f"{predicted_label} {max_prob*100:.2f}%", (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # # Display the image with prediction
                # cv2.imshow('Image Prediction', img)
                # cv2.waitKey(1000)
                # Calculate and print the performance
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Accuracy: {accuracy * 100:.2f}%")


# Calculate and print the performance
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")

sys.stdout = sys.__stdout__

accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"Accuracy: {accuracy * 100:.2f}%")