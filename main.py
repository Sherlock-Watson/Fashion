import os
import pandas as pd
import utils

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
os.makedirs(output_file, exist_ok=True)
path = 'imaterialist-fashion-2020-fgvc7'
label_folder = os.path.join(path, "train_labels")
os.makedirs(label_folder, exist_ok=True)
train_image_folder = os.path.join(path, "train")
csv_path = os.path.join(path, "train.csv")

# image_list = os.listdir(train_image_folder)
# print(image_list)
# df = pd.read_csv(csv_path)
# first_line = df.iloc[0]
# img_id = first_line['ImageId']
# mask_rle = first_line['EncodedPixels']
# height = first_line['Height']
# width = first_line['Width']
# class_id = first_line['ClassId']
# attribute_ids = first_line['AttributesIds']
# img_shape = (width, height)
# print(f"image_id: {img_id}, height: {height}, width: {width}, class id: {class_id}, attribute ids: {attribute_ids}")
# label_path = os.path.join(label_folder, f"{img_id}.txt")
# utils.save_label(label_path, mask_rle, img_shape, class_id)

