import os
import shutil

src = "/media/cai_002/New Volume2/florence2/sku110k/SKU110K_fixed/all"

dest = "/media/cai_002/New Volume2/florence2/sku110k/SKU110K_fixed/labels/"

for file in os.listdir(src):
    print(file)
    shutil.move(f"{src}/{file}", f"{dest}/{file}")