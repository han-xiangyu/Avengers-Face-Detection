import os
import shutil
import random

def split_data(source_dir, target_dir, train_ratio, val_ratio):
    classes = os.listdir(source_dir)
    
    for cls in classes:
        # Create class directories in train, val, and test folders
        os.makedirs(os.path.join(target_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'val', cls), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'test', cls), exist_ok=True)

        # List all files in class directory
        all_files = os.listdir(os.path.join(source_dir, cls))
        all_files = [f for f in all_files if os.path.isfile(os.path.join(source_dir, cls, f))]

        # Shuffle files
        random.shuffle(all_files)

        # Calculate split indices
        total_files = len(all_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        # Split files into train, val, and test sets
        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]

        # Copy files to the new train, val, and test directories
        for f in train_files:
            shutil.copy(os.path.join(source_dir, cls, f), os.path.join(target_dir, 'train', cls, f))
        for f in val_files:
            shutil.copy(os.path.join(source_dir, cls, f), os.path.join(target_dir, 'val', cls, f))
        for f in test_files:
            shutil.copy(os.path.join(source_dir, cls, f), os.path.join(target_dir, 'test', cls, f))

# Usage
source_directory = './dataset/ready_dataset6'
target_directory = './dataset/whole_dataset6'
train_ratio = 0.8  # Training data percentage
val_ratio = 0.1    # Validation data percentage (test_ratio will be 0.1 as well)

split_data(source_directory, target_directory, train_ratio, val_ratio)
