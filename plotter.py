import pandas as pd
import cv2
import os
import time
from tqdm import tqdm

tqdm.pandas()
def plot_boxes(image_path, annotations, save_path):
    # Read the image
    try:
        img = cv2.imread(image_path)
    
        # Plot bounding boxes on the image
        for _, row in annotations.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # classes = row['class']
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # cv2.putText(img, classes, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        # Save the annotated image
        filename = os.path.basename(image_path)
        save_filename = os.path.join(save_path, filename)
        cv2.imwrite(save_filename, img)
    except Exception as e:
        print(f'Cannot open {image_path}', e)

def main(csv_file, image_dir, save_dir):
    # Read annotations from CSV
    annotations = pd.read_csv(csv_file)

    # Iterate over each image
    for _, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        filename = row['imageName']
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path) and not os.path.isfile(image_path):
            print(filename)
            continue
        if os.path.exists(os.path.join(save_dir, filename)) and os.path.isfile(os.path.join(save_dir, filename)):
            # print('PLOTTED')
            continue
        annotations_for_image = annotations[annotations['imageName'] == filename]
        try:
            plot_boxes(image_path, annotations_for_image, save_dir)
        except Exception as e:
            print(f'Cannot open {image_path}', e)
            continue

if __name__ == "__main__":
    start = time.time()
    csv_file = "/media/cai_002/New Volume2/florence2/ph_shelf.csv"
    image_dir = "/media/cai_002/New Volume2/florence2/ph"
    save_dir = "/media/cai_002/New Volume2/florence2/ph_PLOTTED"
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Started plotting .....')
    
    main(csv_file, image_dir, save_dir)
    print('DONE !')
    print('Time taken = ', (time.time()-start))
