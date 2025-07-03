from PIL import Image
import csv
import os

def convert_csv_to_yolo(csv_file, image_folder, output_folder):
    """
    Convert CSV annotation data to YOLO format with dynamic image dimensions.

    Parameters:
    - csv_file (str): Path to the CSV file containing annotations.
    - image_folder (str): Path to the folder containing images.
    - output_folder (str): Path to the output folder to save YOLO files.
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_name = row['filename']
            class_id = int(row['isShelf'])
            xmin, ymin, xmax, ymax = map(float, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])

            # Get image dimensions dynamically
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                print(f"Image {image_name} not found in {image_folder}. Skipping...")
                continue

            with Image.open(image_path) as img:
                img_width, img_height = img.size

            # Calculate YOLO format
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # Format YOLO annotation
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

            # Save to file (one file per image)
            output_file = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.txt")
            with open(output_file, 'a') as out_file:
                out_file.write(yolo_line + '\n')

    print(f"YOLO annotations saved to {output_folder}")

convert_csv_to_yolo('/media/cai_002/New Volume2/florence2/loc_data/anno.csv', '/media/cai_002/New Volume2/florence2/loc_data/imgs'
                    , 'yolo_annotations')
