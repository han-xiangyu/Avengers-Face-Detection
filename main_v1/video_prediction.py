import cv2
import torch
import pickle
from train_model import myNetwork
from helper_functions import *
from yoloface import face_analysis
face_detector = face_analysis()
# Load your trained model
inputSize = 2622
outputSize = 5
hiddenSize1 = 128
hiddenSize2 = 64

model = myNetwork(inputSize, outputSize, hiddenSize1, hiddenSize2)
model.load_state_dict(torch.load('./main_V1/dataset/model_avengers.pth'))

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

# Load the label mapping
with open('./main_V1/dataset/label_mapping_avengers.pkl', 'rb') as f:
    label_dict = pickle.load(f)


# Process a video file
def predict_in_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    # Get video properties for the output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # You can also use 'XVID' or other codecs
    out = cv2.VideoWriter('./output_video.mp4', fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # # Detect, crop, and extract features from the face
        # feature_vector, faces = detect_crop_extract_features(frame)

        # if feature_vector is not None:
        #     feature_tensor = torch.tensor(feature_vector).float().unsqueeze(0)
        #     prediction = predict(feature_tensor, model, data_dict)
        #     print(f"Predicted label: {prediction}")

        ################## Detect faces ##########################
        # face_cascade = cv2.CascadeClassifier('./main_V1//haarcascade_frontalface_default.xml')
        # # Proceed only if the image is loaded successfully
        # if frame is not None:
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # # print(faces)

            # Detect faces and get face bounding boxes
        _, face_boxes, _ = face_detector.face_detection(frame, model='tiny')


        # for i, face in enumerate(faces):
        for (x, y, w, h) in face_boxes:
            cropped_face = frame[y:y+w, x:x+h]

            ## extract feature vector
            if cropped_face is not None:
                # print(cropped_face.shape)
                whole_feature_vector = DeepFace.represent(cropped_face, model_name = 'VGG-Face', enforce_detection=False)
                if whole_feature_vector is not None:
                    feature_vector = (whole_feature_vector[0]['embedding'])

                    feature_tensor = torch.tensor(feature_vector).float().unsqueeze(0)
                    prediction, probability = predict(feature_tensor, model, label_dict)
                    if probability < 0.5:
                        prediction = 'Unknown'
                    cv2.rectangle(frame, (x, y), (x+h, y+w), (255, 0, 0), 2)
                    cv2.putText(frame, f"{prediction} {probability}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        
        
        # Write the frame into the output file
        out.write(frame)

        # Display the frame with prediction
        cv2.imshow('Test Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Call the function with the path to your video
predict_in_video('./dataset/avengers_video.mp4', model)
