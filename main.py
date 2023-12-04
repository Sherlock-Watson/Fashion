import os
import pandas as pd
import utils
from PIL import Image

# Data columns (total 6 columns):
#  #   Column         Non-Null Count   Dtype
# ---  ------         --------------   -----
#  0   ImageId        333401 non-null  object
#  1   EncodedPixels  333401 non-null  object
#  2   Height         333401 non-null  int64
#  3   Width          333401 non-null  int64
#  4   ClassId        333401 non-null  int64
#  5   AttributesIds  206410 non-null  object

output_file = "output"
path = 'imaterialist-fashion-2020-fgvc7'
image_path = os.path.join(path, "train")
csv_path = os.path.join(path, "train.csv")
image_list = os.listdir(image_path)
# print(image_list)
df = pd.read_csv(csv_path)
first_line = df.iloc[0]
img_id = first_line['ImageId']
mask_rle = first_line['EncodedPixels']
height = first_line['Height']
width = first_line['Width']
img_shape = (height, width)
# image_file = Image.open(os.path.join(image_path, f"{img_id}.jpg"))
# print(f"image_shape: {image_file.size}")
print(f"image_id: {img_id}, height: {height}, width: {width}")
decoded_mask = utils.rle_decode(mask_rle, img_shape)
mask_image = Image.fromarray(decoded_mask * 255)
mask_image.save(os.path.join(output_file, f'{img_id}.png'))
# print(decoded_mask)
# print(df.info())