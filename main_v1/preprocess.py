from helper_functions import *
import os

###################### Detect crop and save all images in a directory ################

# # Paths to the directories
# image_directory = './dataset/lfw_dataset'
# processed_directory = './dataset/processed_lfw_dataset'

# # Create the processed directory if it does not exist
# if not os.path.exists(processed_directory):
#     os.makedirs(processed_directory)

# # Iterate through each subfolder
# for person in os.listdir(image_directory):
#     person_path = os.path.join(image_directory, person)
    
#     # Check if it's a directory
#     if os.path.isdir(person_path):
#         # Create a corresponding subfolder in the processed directory
#         processed_person_path = os.path.join(processed_directory, person)
#         if not os.path.exists(processed_person_path):
#             os.makedirs(processed_person_path)

#         # Process each image file in the subfolder
#         for image_file in os.listdir(person_path):
#             image_file_path = os.path.join(person_path, image_file)
            
#             # Define the save path for the cropped image
#             save_path = os.path.join(processed_person_path, image_file)
            
#             # Detect, crop, and save the image
#             detect_crop_image(image_file_path, save_path)


image_directory = './dataset/Chris_Evans'
processed_directory = './dataset/whole_dataset/Chris_Evans'

# Create the processed directory if it does not exist
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

# Iterate through each subfolder
for person in os.listdir(image_directory):
    person_path = os.path.join(image_directory, person)
    
    # Check if it's a directory
    if os.path.isdir(person_path):
        # Create a corresponding subfolder in the processed directory
        processed_person_path = os.path.join(processed_directory, person)
        if not os.path.exists(processed_person_path):
            os.makedirs(processed_person_path)

        # Process each image file in the subfolder
        for image_file in os.listdir(person_path):
            image_file_path = os.path.join(person_path, image_file)
            
            # Define the save path for the cropped image
            save_path = os.path.join(processed_person_path, image_file)
            
            # Detect, crop, and save the image
            detect_crop_image(image_file_path, save_path)