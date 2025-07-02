import cv2
import re
import json
import os
import matplotlib.pyplot as plt

# Paths to your JSON file and image folder
json_path = "/home/cai_002/labelss.json"  # Replace with your JSON file path
image_folder = "/media/cai_002/New Volume2/florence2/SKU-110K Dataset.v1i.florence2-od/test"  # Replace with your images folder path
output_folder = "/media/cai_002/New Volume2/florence2/SKU-110K Dataset.v1i.florence2-od/test_PLOTTED"  # Folder to save images with bounding boxes

os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Function to extract bounding boxes from the "suffix" field
def extract_bounding_boxes(suffix):
    pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    boxes = re.findall(pattern, suffix)
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]

# Load the JSON file
with open(json_path, "r") as f:
    annotations = json.load(f)

# Iterate through the annotations
for annotation in annotations:
    image_name = annotation["image"]
    suffix = annotation["suffix"]

    # Extract bounding boxes
    bounding_boxes = extract_bounding_boxes(suffix)

    # Load the corresponding image
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Draw bounding boxes
    for x1, y1, x2, y2 in bounding_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle in blue

    # Save the image with bounding boxes
    output_path = os.path.join(output_folder, image_name)
    plt.imsave(output_path, image)
    print(f"Saved annotated image: {output_path}")

print("All images processed and saved in:", output_folder)
