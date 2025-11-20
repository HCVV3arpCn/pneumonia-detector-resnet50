import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

show_images = False

# Load and preprocess image (unchanged)
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    img = cv2.resize(img, (224, 224))  # Keep resize for consistency with model
    img = img / 255.0  # Normalize
    return img

# Display images
def display_images(filename, original_img, processed_img):
    # Display original and processed images side by side using matplotlib
    print(f"Creating plot for {filename}...")
    plt.figure(figsize=(10, 5))  # Adjust figure size as needed
    plt.subplot(1, 2, 1)  # First subplot
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Original - {category} - {filename}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)  # Second subplot
    plt.imshow(processed_img * 255, cmap='gray')  # Rescale for display
    plt.title(f"Processed - {category} - {filename}")
    plt.axis('off')
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show(block=True)  # Block until the window is closed
    print(f"Plot for {filename} closed, proceeding to next image...")    

# Data augmentation (unchanged)
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# Use absolute path based on your current directory
base_path = "/Users/jrosenst/Repositories_personal/chestx"
dataset_path = os.path.join(base_path, "dataset/chest_xray/train")
preprocessed_data = []

# Loop through training data and display original and processed images
for category in ["NORMAL", "PNEUMONIA"]:
    category_path = os.path.join(dataset_path, category)
    if not os.path.exists(category_path):
        print(f"Directory not found: {category_path}")
        continue
    for filename in os.listdir(category_path):
        print(f"Preprocessing {filename}...")
        if filename.lower().endswith((".jpg", ".jpeg")):
            image_path = os.path.join(category_path, filename)
            
            # Read original image
            original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                print(f"Failed to load original image: {image_path}")
                continue
            
            # Preprocess image
            processed_img = preprocess_image(image_path)
            if processed_img is None:
                continue

            # Display images
            if show_images:
                display_images(filename, original_img, processed_img)
            
            # Append to preprocessed data
            preprocessed_data.append(processed_img)

preprocessed_data = np.array(preprocessed_data) if preprocessed_data else np.array([])

import code; code.interact(local=locals())