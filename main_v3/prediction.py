# Prediction
# Author: Xiangyu Han
# Date: 2023.12
import cv2
import torch
from helper_functions import *
import torchvision.models as models

data_dir = './dataset/whole_dataset5'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Create an instance of the VGG16 model
# If you have modified the classifier part of the VGG model, ensure to replicate those changes here
myVGG16 = models.vgg16(pretrained=False)
# Modifying the Classifier
num_ftrs = myVGG16.classifier[6].in_features
myVGG16.classifier[6] = nn.Linear(num_ftrs, len(class_names))

# Load the trained model parameters from a file
myVGG16.load_state_dict(torch.load('./models/model_avengers_V5.1.pth'))
myVGG16.eval()

# Move the model to GPU if available
if torch.cuda.is_available():
    myVGG16 = myVGG16.cuda()
    
image_path = './dataset/avengers.jpg'
img = cv2.imread(image_path)

# Use the save_detected_face function to get detected faces and their coordinates
detected_faces, faces_coordinates = detect_face(img)

# Loop over each detected face for prediction
for i, face in enumerate(detected_faces):
    max_prob, predicted_label = predict(face, myVGG16,class_names)
    x, y, w, h = faces_coordinates[i]
    print(predicted_label,max_prob)
    # cv2.imshow('Croped face', face)
    # cv2.waitKey(0)
    # Draw rectangles and labels on the original image
    cv2.rectangle(img, (x, y), (x + h, y + w), (255, 0, 0), 2)
    cv2.putText(img, f"Label: {predicted_label}, Probability: {max_prob*100}%", (x-200, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with prediction
cv2.imshow('Image Prediction', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
