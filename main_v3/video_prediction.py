import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from helper_functions import *

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

# Function for video prediction
def predict_in_video(video_path, myResnet):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter('./output_video.mp4', fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_faces, faces_coordinates = detect_face(frame)

        if detected_faces is not None:
            for i, face in enumerate(detected_faces):
                max_prob, predicted_label = predict(face, myResnet, class_names)
                x, y, w, h = faces_coordinates[i]

                if max_prob < 0.7:
                    predicted_label = 'Unknown'

                cv2.rectangle(frame, (x, y), (x + h, y + w), (255, 0, 0), 2)
                cv2.putText(frame, f"{predicted_label} {max_prob*100:.2f}%", (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Test Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == '__main__':
    # Call functions or include additional code as needed
    predict_in_video('./dataset/avengers_video.mp4', myResNet)
