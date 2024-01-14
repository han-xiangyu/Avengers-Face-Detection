import os
from helper_functions import *
import pickle

##########################  Extract the features  ###########################
# Path to your dataset directories
# dataset_paths = ['./dataset/processed_lfw_dataset']
dataset_paths = ['./dataset/test_lfw_RDJ']

label_mapping = {}
current_label = 0

for dataset_path in dataset_paths:
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path) and person not in label_mapping:
            label_mapping[person] = current_label
            current_label += 1

# Pre-trained model setup
model = 'VGG-Face'

# Dictionary to store feature vectors indexed by numeric labels
feature_vectors_by_label = {}

for dataset_path in dataset_paths:
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            person_label = label_mapping[person]
            person_features = []
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                print(f"Processing {image_path}")  # Debugging statement
                features = extract_features(image_path, model)
                if features is not None:
                    person_features.append(features[0]['embedding'])
                else:
                    # Optionally, remove the label from the label mapping if no features were valid
                    del label_mapping[person]
            feature_vectors_by_label[person_label] = person_features



# Save feature vectors by label
with open('./dataset/feature_vectors_by_label.pkl', 'wb') as f:
    pickle.dump(feature_vectors_by_label, f)

# Save label mapping
with open('./dataset/label_mapping.pkl', 'wb') as f:
    pickle.dump(label_mapping, f)

# print('The first_pair is ', list(feature_vectors_by_label.values())[:2])
# print('The num of pair of feature_vectors_by_label is ',len(feature_vectors_by_label))
# print('The num of pair of label_mapping is ',len(label_mapping))
# print(len(list(feature_vectors_by_label.values())[3]))
