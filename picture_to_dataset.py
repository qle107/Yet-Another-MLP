import os
from PIL import Image
import numpy as np
import pandas as pd

# Paths to the folders
image_folder = './DATA/train/'
folders = ['plane', 'car', 'bike']

# Image size
image_size = (128, 128)

# Initialize lists to hold image data and labels
data = []
labels = []

# Loop through each folder and process the images
for label, folder in enumerate(folders):
    folder_path = os.path.join(image_folder, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                # Resize the image
                img_resized = img.convert('L').resize(image_size)

                # CNN


                # End of CNN
                # Convert image to numpy array and flatten it
                img_array = np.array(img_resized).flatten()
                # Append the flattened image and label to the lists
                data.append(img_array)
                labels.append(label)

# Convert the lists to numpy arrays
data_array = np.array(data)
labels_array = np.array(labels)

# Create a DataFrame
dataset = pd.DataFrame(data_array)
dataset['label'] = labels_array

# Save the dataset to a CSV file
dataset.to_csv('128_dataset.csv', index=False)

print("Dataset created and saved as 'another_image_dataset.csv'")