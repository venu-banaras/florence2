import os
import pandas as pd
from PIL import Image
imgs = os.listdir("/media/cai_002/New Volume2/florence2/ph")
df = pd.read_csv("/media/cai_002/New Volume2/florence2/ph_shelf.csv")
os.makedirs("/media/cai_002/New Volume2/florence2/ph_shelf", exist_ok=True)


for index, row in df.iterrows():
    image = row['imageName']
    if image in imgs:
        print(f"{image} in folder. CROPPING")
        box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        print(box)
        img = Image.open(f"/media/cai_002/New Volume2/florence2/ph/{image}").convert("RGB")
        cropped = img.crop(box)
        try:
            cropped.save(f"/media/cai_002/New Volume2/florence2/ph_shelf/{index}_{image}")
        except Exception as e:
            print(e)

