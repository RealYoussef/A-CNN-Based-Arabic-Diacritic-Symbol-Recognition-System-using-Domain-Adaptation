import os
import csv
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance
import random
import glob
import numpy as np
import json
import math
import cv2
from skimage.filters import threshold_otsu


# Define the Arabic letters and their Unicode equivalents
letters = [
    ('\uFE76', 0), # Fatha
    ('\uFE78', 1), # Damma
    ('\uFE7A', 2), # Kasra
    ('\uFE70', 3), # Fathatan
    ('\uFE72', 4), # Dammatan
    ('\uFE74', 5), # Kasratan
    ('\uFE7C', 6), # Shadda
    ('\uFE7E', 7), # Sukun
    ('\uFC63', 8), # Shadda with Alef
    ('\uFC60', 9), # Shadda with Fatha
    ('\uFC61', 10), # Shadda with Damma
    ('\uFC62', 11), # Shadda with Kasra
    ('\uFC5E', 12), # Shadda with Dammatan
    ('\uFC5F', 13) # Shadda with Kasratan
]
variations = 20

def calculate_bbox_from_mask(mask):
    if np.count_nonzero(mask) == 0:
        return (0, 0, 0, 0)
    else:
        nonzero_indices = np.nonzero(mask)
        y_min = np.min(nonzero_indices[0])
        y_max = np.max(nonzero_indices[0])
        x_min = np.min(nonzero_indices[1])
        x_max = np.max(nonzero_indices[1])
        return (x_min, y_min, x_max, y_max)


# Define the fonts folder
fonts_folder = 'fonts'

# Get all font files with .ttf or .otf extension in the fonts folder
font_files = glob.glob(os.path.join(fonts_folder, '*.ttf')) + glob.glob(os.path.join(fonts_folder, '*.otf'))

# Sort the font files alphabetically
font_files.sort()

# Create the output folders if they don't exist
os.makedirs('images', exist_ok=True)
os.makedirs('labels', exist_ok=True)

# Delete all files in the 'images' folder
image_files = glob.glob('images/*')
for file in image_files:
    os.remove(file)

# Delete all files in the 'labels' folder
label_files = glob.glob('labels/*')
for file in label_files:
    os.remove(file)

# Create a CSV file for storing the labels
csv_file = open('labels/labels.csv', 'w', newline='', encoding='utf-16')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Image', 'Label', 'X', 'Y', 'Width', 'Height'])

# Set the desired image size
image_size = (128, 128)

# Generate and save the images with labels
for i, font_file in enumerate(font_files):
    font_name = f'Font{i+1}'
    # font_name = os.path.splitext(os.path.basename(font_file))[0]  # Get the filename without extension

    try:
        # Load the font to check if it supports UTF-16
        font = ImageFont.truetype(font_file, 1)
    except OSError:
        # Skip the font if it does not support UTF-16
        continue

    # Create a blank image with the specified size and background color
    background_color = (0)  # Black background color
    image = Image.new('L', image_size, background_color)

    # Define the font properties
    if i < 4:
        font_size = 70  # For the first two fonts, use a font size of 70
    elif i < 50:
        font_size = 140  # For the 3rd to 40th fonts, use a font size of 140
    else:
        font_size = 70  # For fonts after the 40th font, use a font size of 70

    font_color = (255)  # White text color
    font_color_range = (10, 255)  # Initialize as reverse range

    for letter, unicode_name in letters:
        for j in range(variations):
            # Clear the previous content of the image
            draw = ImageDraw.Draw(image)
            draw.rectangle([(0, 0), image_size], fill=background_color)

            # Load the font
            font = ImageFont.truetype(font_file, font_size)

            # Calculate the position to center the text
            text_width, text_height = draw.textsize(letter, font=font)
            text_x = (image_size[0] - text_width) // 2
            text_y = (image_size[1] - text_height) // 2

            # Draw the deformed text on the image
            draw.text((text_x, text_y), letter, font=font, fill=font_color)
            
            # Apply perspective distortions
            perspective_distortion = random.uniform(0, 0.05)  # Adjust the perspective distortion factor as desired
            
            matrix2 = (
                1 + random.uniform(-perspective_distortion, perspective_distortion),
                random.uniform(-perspective_distortion, perspective_distortion),
                random.uniform(-perspective_distortion, perspective_distortion),
                random.uniform(-perspective_distortion, perspective_distortion),
                1 + random.uniform(-perspective_distortion, perspective_distortion),
                random.uniform(-perspective_distortion, perspective_distortion),
                0, 0
            )
            image = image.transform(image_size, Image.PERSPECTIVE, matrix2, Image.BICUBIC)
            
            # Apply slant variations
            slant_angle = random.uniform(-5, 5)  # Adjust the slant angle as desired
            slant_radians = math.radians(slant_angle)
            matrix3 = (
                1, slant_radians, 0,
                0, 1, 0,
                0, 0
            )
            image = image.transform(image_size, Image.PERSPECTIVE, matrix3, Image.BICUBIC)
            
            matrix = (
                1 + random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05),
                random.uniform(-0.05, 0.05), 1 + random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05),
                random.uniform(-0.0001, 0.0001), random.uniform(-0.0001, 0.0001)
                )

            # Apply random distortions, skewness, and blur
            image = image.rotate(random.uniform(-3, 3), resample=Image.BILINEAR)
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.2)))
            image = image.transform(image_size, Image.PERSPECTIVE, matrix, Image.BICUBIC)
            
            # Apply illumination variations
            # enhancer = ImageEnhance.Brightness(image)
            # brightness_factor = random.uniform(0.8, 2.5)  # Adjust the brightness factor as desired
            # image = enhancer.enhance(brightness_factor)
            
            # enhancer = ImageEnhance.Contrast(image)
            # contrast_factor = random.uniform(0.8, 1.2)  # Adjust the contrast factor as desired
            # image = enhancer.enhance(contrast_factor)
            
            # enhancer = ImageEnhance.Color(image)
            # color_factor = random.uniform(0.8, 1.2)  # Adjust the color factor as desired
            # image = enhancer.enhance(color_factor)
            
            # Define the film grain parameters
            grain_intensity = random.uniform(0, 10) # Adjust the intensity of the grain
            grain_size = random.randint(1, 6)  # Adjust the size of the grains (Default is 1)

            # Generate film grain
            noise = np.random.normal(0, grain_intensity, (image_size[0] // grain_size, image_size[1] // grain_size))
            noise = np.repeat(np.repeat(noise, grain_size, axis=0), grain_size, axis=1)
            noise = np.clip(noise, 0, 255).astype(np.uint8)
            noise_image = Image.fromarray(noise, mode='L')
            noise_image = noise_image.resize(image_size)

            # Overlay the film grain onto the image
            image = Image.blend(image, noise_image.convert('L'), alpha=0.2)
            
            # Create a binary mask based on the font color range
            mask = np.zeros(image_size, dtype=np.uint8)
            image_array = np.array(image)  # Convert the PIL Image to a NumPy array
            font_color_range_mask = (image_array >= font_color_range[0]) & (image_array <= font_color_range[1])
            mask[font_color_range_mask] = 1

            mask = np.array(mask)
            transformed_bbox = calculate_bbox_from_mask(mask)
            
            # Generate a random quality level between 30 and 70
            quality_level = random.randint(50, 100)
            
            # Save the image with random JPEG compression artifacts
            temp_image_file = f'images/{font_name}_{unicode_name}_{j}_temp.jpg'  # Save as temporary JPEG file
            image.save(temp_image_file, format='JPEG', quality=quality_level)
            
            # Open the temporary JPEG file and convert it back to PNG format
            temp_image = Image.open(temp_image_file)
            image_file = f'images/{font_name}_{unicode_name}_{j}.png'  # Save as PNG file
            temp_image.save(image_file, format='PNG')
            
            # Delete the temporary JPEG file
            os.remove(temp_image_file)
            
            # Remove the "images/" part from the image file path
            image_file = image_file.replace('images/', '')

            # Write the label to the CSV file
            csv_writer.writerow([image_file, unicode_name, *transformed_bbox])
            
            # Create a JSON file for the mask
            mask_file = f'labels/{font_name}_{unicode_name}_{j}.json'
            mask_data = {'mask': mask.tolist()}

            # Save the mask JSON file
            with open(mask_file, 'w') as f:
                json.dump(mask_data, f)

csv_file.close()
