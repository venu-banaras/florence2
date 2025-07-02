import pandas as pd
import os
import cv2
from tqdm import tqdm

anno_folder = "/media/cai_002/New Volume2/florence2/test"
img_folder = "/media/cai_002/New Volume2/dgx_data/vicks_qm/yolo/test"


df = pd.DataFrame()

img_name = []
class_id = []
xmin = []
ymin = []
xmax = []
ymax = []
for file in tqdm(sorted(os.listdir(anno_folder))):
    img_file = file.split(".txt")[0]
    img_file = img_file + ".jpg"
    # print(img_file)

    img = cv2.imread(f"{img_folder}/{img_file}")
    h,w,_ = img.shape

    with open(f"{anno_folder}/{file}", "r") as fp:
        for line in fp:
            # print(type(line))
            lst = str.split(line)
            # print(lst)
            clss, x_center, y_center, width, height = lst
            x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
            # print(type(x_center), y_center, width, height)
            x_center = float(x_center * w)
            y_center = float(y_center * h)
            width = float(width * w)
            height = float(height * h)
        

            xmn = int(x_center - width / 2)
            ymn = int(y_center - height / 2)
            xmx = int(x_center + width / 2)
            ymx = int(y_center + height / 2)

            img_name.append(img_file)
            class_id.append(clss)
            xmin.append(xmn)
            ymin.append(ymn)
            xmax.append(xmx)
            ymax.append(ymx)

df["Image Name"] = img_name
df["class"] = class_id
df["xmin"] = xmin
df["ymin"] = ymin
df["xmax"] = xmax
df["ymax"] = ymax

df.to_csv("/media/cai_002/New Volume2/dgx_data/general_loc/test.csv", index=False)
            