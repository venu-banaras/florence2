import os
import json
from PIL import Image, ImageDraw

def plot_bboxes_from_json(json_path, image_folder, output_folder):
    """
    Plots bounding boxes on images as per the JSON file and saves the output.

    Parameters:
    - json_path (str): Path to the JSON file containing image names and bounding boxes.
    - image_folder (str): Path to the folder containing the images.
    - output_folder (str): Path to the folder where the output images will be saved.
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    for image_name, bboxes in data.items():
        image_path = os.path.join(image_folder, image_name)
        
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found in {image_folder}. Skipping...")
            continue
        
        # Open the image
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # Draw each bounding box
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Save the output image
        output_path = os.path.join(output_folder, image_name)
        image.save(output_path)
        print(f"Processed {image_name}, saved to {output_path}")

# Example usage
json_path = "/raid/training_data/ph_live/ph_live_loc.json"          # Path to your JSON file
image_folder = "/raid/training_data/ph_live/images"         # Path to your image folder
output_folder = "/raid/training_data/ph_live/output_folder"       # Path to save output images

plot_bboxes_from_json(json_path, image_folder, output_folder)
