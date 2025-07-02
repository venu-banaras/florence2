# import pandas as pd
# import os
# import cv2
# from tqdm import tqdm
# import json

# anno_folder = "labels"
# img_folder = "images"

# anno_json = []

# for file in tqdm(sorted(os.listdir(anno_folder))):
#     img_file = file.split(".txt")[0]
#     img_file = img_file + ".jpg"
#     # print(img_file)
#     prefix = "<OD>"
#     suffix_lines = []

#     img = cv2.imread(f"{img_folder}/{img_file}")
#     h,w,_ = img.shape

#     with open(f"{anno_folder}/{file}", "r") as fp:
#         for line in fp:
#             # print(type(line))
#             lst = str.split(line)
#             # print(lst)
#             clss, x_center, y_center, width, height = lst
#             x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
#             # print(type(x_center), y_center, width, height)

#             xmn = int((x_center - width / 2)*1000)
#             ymn = int((y_center - height / 2)*1000)
#             xmx = int((x_center + width / 2)*1000)
#             ymx = int((y_center + height / 2)*1000)
#             clss = int(clss)
#             if clss == 0:
#                 clss = 'SKU'
            
#             suffix = f"{clss}<loc_{xmn}><loc_{ymn}><loc_{xmx}><loc_{ymx}>"
#             suffix_lines.append(suffix)
#         jsons = {
#             "image": img_file,
#             "prefix": prefix,
#             "suffix": "".join(suffix_lines)
#         }
#     jsons_str = json.dumps(jsons, separators=(',', ':'))
#     anno_json.append(jsons_str)

# with open(f"labels.json", "w") as fp:
#     fp.write("\n".join(anno_json))
import os
import json
from tqdm import tqdm

# annotations_dir = "/content/objectdetection-5/train/labels"
# output_json_file = "/content/objectdetection-5/train/images/train_annotations.json"


# annotations_dir = "/content/objectdetection-5/test/labels"
# output_json_file = "/content/objectdetection-5/test/images/test_annotations.json"

annotations_dir = "/media/cai_002/New Volume2/florence2/loc_data/yolo_annotations"
output_json_file = "/media/cai_002/New Volume2/florence2/loc_data/phsku110k.json"



def parse_yolo_annotation(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    image_name = os.path.basename(annotation_file).replace('.txt', '.jpg')
    prefix = "<OD>"
    suffix_lines = []

    for line in lines:
        parts = line.strip().split()
        class_name = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x1 = int((x_center - width/2) * 1000)
        y1 = int((y_center - height/2) * 1000)
        x2 = int((x_center + width/2) * 1000)
        y2 = int((y_center + height/2) * 1000)

        # Replace '0' with 'face' in the class name if it equals '0'
        if class_name == '0':
            class_name = 'SKU'
        # elif class_name == '1':
        #     class_name = 'Digital Multimeter'
        # elif class_name == '2':
        #     class_name = 'Digital Trainer'
        # elif class_name == '3':
        #     class_name = 'Function Generator'
        # elif class_name == '4':
        #     class_name = 'Oscilloscope'
        suffix_line = f"{class_name}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
        suffix_lines.append(suffix_line)

    json_obj = {
        
        "prefix": prefix,
        "suffix": "".join(suffix_lines),
        "image": image_name
    }

    return json_obj

annotations_json_strings = []

for filename in tqdm(sorted(os.listdir(annotations_dir))):
    if filename.endswith(".txt"):
        annotation_file = os.path.join(annotations_dir, filename)
        annotation_obj = parse_yolo_annotation(annotation_file)
        json_string = json.dumps(annotation_obj, separators=(',', ':'))
        annotations_json_strings.append(json_string)

with open(output_json_file, 'w') as json_file:
    json_file.write("\n".join(annotations_json_strings))

print(f"Annotations have been written to {output_json_file}")