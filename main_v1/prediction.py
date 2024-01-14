import torch
from train_model import myNetwork  # Make sure to import your model's class
from helper_functions import *

# Image path
# image_path = './dataset/processed_lfw_dataset/Aaron_Pena/Aaron_Pena_0001.jpg'
image_path = './10085.jpg'
# Define your model's architecture (make sure it's the same as used for training)
inputSize = 2622
outputSize = 5749  # Number of classes
hiddenSize1 = 128
hiddenSize2 = 64

model = myNetwork(inputSize, outputSize, hiddenSize1, hiddenSize2)

# Load the trained weights
model.load_state_dict(torch.load('modelRDJ.pth'))



# If you have a GPU, move the model to GPU
if torch.cuda.is_available():
    model = model.cuda()

# predict single image
img = cv2.imread(image_path)
prediction = predict(img, model)
print(f"The label is :{prediction}")
