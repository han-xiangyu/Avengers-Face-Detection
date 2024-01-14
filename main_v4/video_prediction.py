import os
import cv2
import torch
import pickle
from facenet_pytorch import InceptionResnetV1
from helper_functions import *
from yoloface import face_analysis

# Load the model for image prediction
data_dir = './dataset/whole_dataset6'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myResnet = InceptionResnetV1(
    classify=True,
    num_classes=len(class_names)).to(device)
myResnet.load_state_dict(torch.load('./models/model_avengers_ResNet_V3.pth'))
myResnet.eval()


face_detector = face_analysis()

def predict_in_video(video_path, myResnet):
    cap = cv2.VideoCapture(video_path)

    # Get video properties for the output video
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
        
        # Write the frame into the output file
        out.write(frame)

        # Display the frame with prediction
        cv2.imshow('Test Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Call the function with the path to your video
predict_in_video('./dataset/avengers_video.mp4', myResnet)
