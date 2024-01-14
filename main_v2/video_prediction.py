import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from helper_functions import *

# Load the model
data_dir = './dataset/whole_dataset5'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

myVGG16 = models.vgg16(pretrained=False)
num_ftrs = myVGG16.classifier[6].in_features
myVGG16.classifier[6] = nn.Linear(num_ftrs, len(class_names))
myVGG16.load_state_dict(torch.load('./models/model_avengers_V5.1.pth'))
myVGG16.eval()

if torch.cuda.is_available():
    myVGG16 = myVGG16.cuda()

# Load and configure myResNet for video prediction (using your training code)
# Place your model training code here

# Function for video prediction
def predict_in_video(video_path, myVGG16):
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
                max_prob, predicted_label = predict(face, myVGG16, class_names)
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
    predict_in_video('./dataset/avengers_video.mp4', myVGG16)
