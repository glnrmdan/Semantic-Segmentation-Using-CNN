import os
import cv2
import numpy as np

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Convert BGR to RGB and append to the list
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images

def load_images_from_folder_filenames(folder):
    images = []
    filenames = []  # List to store filenames
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Convert BGR to RGB and append to the list
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            filenames.append(filename)  # Append filename
    return images, filenames

# Function to load ground truth labels from a directory
def load_ground_truth_from_folder(folder):
    ground_truths = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ground_truths.append(img_rgb)
    return ground_truths

def prepare_data(feature_maps, ground_truths):
    X = []  # Features
    y = []  # Labels
    
    for feature_map, ground_truth in zip(feature_maps, ground_truths):
        height, width, _ = feature_map.shape
        for h in range(height):
            for w in range(width):
                # Flatten the feature map for each pixel
                features = feature_map[h, w, :].flatten()
                X.append(features)
                
                # Check ground truth to assign label
                if np.array_equal(ground_truth[h, w], [0, 255, 255]) or np.array_equal(ground_truth[h, w], [0, 255, 0]):
                    label = 0  # Non-vegetation
                else:
                    label = 1  # vegetation
                    
                # Convert label to one-hot encoding
                y.append([1, 0] if label == 1 else [0, 1])
    
    return np.array(X).T, np.array(y).T