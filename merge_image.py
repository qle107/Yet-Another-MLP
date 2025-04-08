# import os
# import shutil
# import random
#
# # Path to the main directory containing the car folders
# source_dir = './DATA/train'
# # Path to the new directory where random images will be copied
# dest_dir = './DATA/car'
# target_image_count = 2000
#
# # Create the destination directory if it doesn't exist
# os.makedirs(dest_dir, exist_ok=True)
#
# # Function to get all images from the subfolders
# def get_all_images(source_dir):
#     image_paths = []
#     for subdir in os.listdir(source_dir):
#         subdir_path = os.path.join(source_dir, subdir)
#         if os.path.isdir(subdir_path):
#             # List all files in the current subfolder
#             files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
#             for file in files:
#                 image_paths.append((subdir, file))
#     return image_paths
#
# # Get all image paths from the source directory
# all_images = get_all_images(source_dir)
#
# # Shuffle the list to randomize the selection
# random.shuffle(all_images)
#
# # Set to keep track of selected images to avoid duplicates
# selected_images = set()
#
# # Copy images until the target count is reached
# for subdir, file in all_images:
#     if len(selected_images) >= target_image_count:
#         break
#     source_file = os.path.join(source_dir, subdir, file)
#     dest_file = os.path.join(dest_dir, f"{subdir}_{file}")
#
#     if dest_file not in selected_images:
#         # Ensure the destination directory exists
#         os.makedirs(os.path.dirname(dest_file), exist_ok=True)
#         shutil.copy(source_file, dest_file)
#         selected_images.add(dest_file)
#
# print(f"{len(selected_images)} images have been copied to the 'car' directory.")


# import os
# import re
#
# # Define the directory containing your files
# directory = 'C:\\Users\\annan\\Desktop\\ESGI\\3rd year\\LR_MLP_Working\\DATA\\bike'
#
# # Change the working directory to the folder with the files
# os.chdir(directory)
#
# # Iterate over all files in the directory
# for filename in os.listdir(directory):
#     # Use a regular expression to match filenames like 'Bike (1).jpeg', 'Bike (2).jpeg', etc.
#     match = re.match(r'Bike \((\d+)\)\.JPG', filename)
#     if match:
#         # Extract the number from the filename
#         number = match.group(1)
#         # Construct the new filename
#         new_filename = f'bike_{number}.jpg'
#         # Rename the file
#         os.rename(filename, new_filename)
#
# print("Renaming complete.")

import os

# Define the path to the directory containing the images
directory = './DATA/train/plane'  # Update this path to match the location of your files

# List all files in the directory
files = os.listdir(directory)

# Loop through all files in the directory
for i, filename in enumerate(files):
    if filename.startswith('TF_PLANE_') and (filename.endswith('.jpg') or filename.endswith('.png')):
        # Construct the new filename
        new_filename = f'plane_{i + 1}.jpg'
        # Construct full file path
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')

