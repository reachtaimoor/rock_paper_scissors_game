import cv2
import os

# Define original and new directories for images
original_directories = ['rock_images', 'paper_images', 'scissors_images']
new_directories = ['rock_images_cropped', 'paper_images_cropped', 'scissors_images_cropped']

# Create new directories if they don't exist
for new_dir in new_directories:
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

# Function to load all images from a directory, crop the top, and save them to a new directory
def crop_top_of_images(old_directory, new_directory, crop_height):
    images = os.listdir(old_directory)
    for image_name in images:
        image_path = os.path.join(old_directory, image_name)
        image = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if image is not None:
            # Crop the top portion of the image
            cropped_image = image[crop_height:, :]  # Remove the top 'crop_height' pixels

            # Save the cropped image to the new directory
            new_image_path = os.path.join(new_directory, image_name)
            cv2.imwrite(new_image_path, cropped_image)
        else:
            print(f"Failed to load image: {image_path}")

# Define the height to crop from the top of the image
crop_height = 40  # Adjust this value as needed

# Process all images in each directory
for old_dir, new_dir in zip(original_directories, new_directories):
    crop_top_of_images(old_dir, new_dir, crop_height)

# Notify completion
print("Image cropping completed for all images in the specified directories.")