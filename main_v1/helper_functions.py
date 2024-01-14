# Functions used to preprocess images
import os
import cv2
from deepface import DeepFace
import torch
from helper_functions import *
import torch.nn.functional as F

# Face detection and crop function
def detect_crop_image(image_path, save_path):
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)

    # Proceed only if the image is loaded successfully
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            
            # Save the cropped image
            cv2.imwrite(save_path, cropped_face)


# Function to create a mapping from person's name to a unique label
def create_label_mapping(directory):
    """
    Create a mapping from person's name to a unique label.
    :param directory: Path to the LFW dataset directory.
    :return: Dictionary mapping each person to a unique label.
    """
    # List all folders/persons in the directory
    persons = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    # Create a mapping from person's name to a unique number
    label_mapping = {person: idx for idx, person in enumerate(persons)}
    # label_mapping now contains something like {'person_name_1': 0, 'person_name_2': 1, ...}
    return label_mapping
    
def extract_features(image_path,model='VGG-Face'):
    """
    Extract features from an image.
    :param image_path: Path to the image.
    :return: Feature vector.
    """
    # Load the image
    img = cv2.imread(image_path)
    # Extract features from the image
    try:
        representation = DeepFace.represent(img, model_name = model, enforce_detection=False)
        # Proceed with storing features and labels
    except ValueError as e:
        print(f"Warning: {e} - Image Path: {image_path}")
        # Optionally, log or skip this image
        return None
    return representation

def detect_crop_extract_features(img, model='VGG-Face'):
    face_cascade = cv2.CascadeClassifier('./main_V1//haarcascade_frontalface_default.xml')

    # Initialize cropped_face
    cropped_face = None

    # Proceed only if the image is loaded successfully
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            # break  # Assuming you only need the first detected face

    if cropped_face is not None:
        try:
            # print(cropped_face.shape)
            whole_feature_vector = DeepFace.represent(cropped_face, model_name = model, enforce_detection=False)
            if whole_feature_vector is not None:
                feature_vector = (whole_feature_vector[0]['embedding'])
                return feature_vector, faces
        except ValueError as e:
            print(f"Warning: {e} - This image cannot be extract feature vectors")
    else:
        print("No faces detected in the image")

    return None


################ Predict the results #################
def predict(feature_vector, model, label_dict):
    # Reverse the dictionary to map numbers to strings
    reversed_dict = {value: key for key, value in label_dict.items()}

    # Define a loss function (cross-entropy loss)
    criterion = torch.nn.CrossEntropyLoss()

    # Convert to PyTorch tensor and add a batch dimension
    feature_tensor = torch.tensor(feature_vector).float().unsqueeze(0)
    
    # Move tensor to GPU if available
    if torch.cuda.is_available():
        feature_tensor = feature_tensor.cuda()

    # Set the model to evaluation mode
    model.eval()

    # No need to track gradients for prediction
    with torch.no_grad():
        # Make the prediction
        output = model(feature_tensor)
        # print(output.shape)  # Add this line to debug

        output = torch.squeeze(output)  # This will change the shape from [1, 1, 5749] to [1, 5749]

        # # Process the output, applying a argmax
        # preds.append(torch.argmax(outputs, dim=-1).cpu().detach().numpy())
        # Use argmax to get the predicted label (index of the highest value)
        numeric_label  = torch.argmax(output, dim=0)


                # Make the prediction
        logits = model(feature_tensor)
        logits = torch.squeeze(logits)
        probabilities = F.softmax(logits, dim=0)
        max_prob, numeric_label = torch.max(probabilities, dim=0)

        # Map the numeric label to string using the reversed dictionary
        string_label = reversed_dict.get(numeric_label.item())
    
    return string_label, max_prob.item()