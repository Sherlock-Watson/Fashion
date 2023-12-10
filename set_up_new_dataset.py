import pandas as pd
import os
import shutil

import utils
from collections import Counter
import json

# Data columns (total 6 columns):
#  #   Column         Non-Null Count   Dtype
# ---  ------         --------------   -----
#  0   ImageId        333401 non-null  object
#  1   EncodedPixels  333401 non-null  object
#  2   Height         333401 non-null  int64
#  3   Width          333401 non-null  int64
#  4   ClassId        333401 non-null  int64
#  5   AttributesIds  206410 non-null  object


def create_folders():
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(new_train_image_folder, exist_ok=True)
    os.makedirs(new_train_label_folder, exist_ok=True)
    os.makedirs(new_test_image_folder, exist_ok=True)
    os.makedirs(new_test_label_folder, exist_ok=True)
    os.makedirs(new_val_image_folder, exist_ok=True)
    os.makedirs(new_val_label_folder, exist_ok=True)

def generate_yaml_string(path):
    with open(path) as json_file:
        label_description = json.load(json_file)
    categories = label_description['root']['categories']
    string_list = []
    for category in categories:
        string_list.append(f"{category['id']: {category['name']}}")
    return "\n  " + "\n  ".join(string_list)


with open("paths.json") as file:
    paths = json.load(file)

old_csv_path = paths['old_csv_path']
new_csv_path = paths['new_csv_path']

# set up the new dataset folder
dataset_folder = paths['dataset']
new_train_image_folder = paths['new_train_images']
new_train_label_folder = paths['new_train_labels']
new_test_image_folder = paths['new_test_images']
new_test_label_folder = paths['new_test_labels']
new_val_image_folder = paths['new_val_images']
new_val_label_folder = paths['new_val_labels']
create_folders()

whole_df = pd.read_csv(old_csv_path)
df = whole_df[['ImageId', 'ClassId']]
# get original distribution
class_ids = df['ClassId']
element_counts = Counter(class_ids)

threshold = 500

# get the classes with few samples
minority_class_ids = []
for element, count in element_counts.items():
    if count > threshold:
        continue
    minority_class_ids.append(element)
print(f"The minority class ids are: {minority_class_ids}")

# get the related image files
file_set = set()
for class_id in minority_class_ids:
    sub_file_set = set(df[df['ClassId'] == class_id]['ImageId'])
    file_set = file_set.union(sub_file_set)
print(f"size of current file list: {len(file_set)}")

# get the new distribution
sub_df = df[df['ImageId'].isin(file_set)]
element_counts = Counter(sub_df['ClassId'])

# There is still some classes with few samples (below threshold)
need_supply = []
for element, count in element_counts.items():
    if count > threshold or element in minority_class_ids:
        continue
    need_supply.append(element)
print(f"These class ids need more samples: {need_supply}")

# get more samples according to the need_supply dict
for class_id in need_supply:
    sub_file_list = df[df['ClassId'] == class_id]['ImageId']
    # number of entries
    count = 0
    sub_file_set = set()
    amount = threshold - df[(df['ClassId'] == class_id) & (df['ImageId'].isin(file_set))].shape[0]
    if amount <= 0:  # Enough samples
        continue
    for img_id in sub_file_list:
        if img_id in file_set:
            # skip the images that has been selected
            continue
        count += 1
        sub_file_set.add(img_id)
        if count >= amount:  # enough samples
            break
    file_set = file_set.union(sub_file_set)
print(f"Image count: {len(file_set)}")
sub_df = whole_df[df['ImageId'].isin(file_set)]
utils.print_class_id_distribution(sub_df, 'ClassId')
print(f"size of new df: {sub_df.shape[0]}")
sub_df.to_csv(new_csv_path, index=False)



# configuration
# label_description_path = "imaterialist-fashion-2020-fgvc7/label_descriptions.json"
# category_info = generate_yaml_string(label_description_path)
# content = f"""
# # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
# path: dataset  # dataset root dir
# train: {new_train_image_folder}  # train images (relative to 'path') 4 images
# val: {new_val_image_folder}  # val images (relative to 'path') 4 images
# test: {new_test_image_folder}
#
# # Classes (80 COCO classes)
# names:
# {category_info}
# """
# with open("dataset/fashion_dataset.yaml") as file:
#     file.write(content)

# paste all the train images involved
# for img_id in file_set:
#     shutil.copyfile(f"imaterialist-fashion-2020-fgvc7/train/{img_id}.jpg",
#                     os.path.join(new_train_image_folder, f"{img_id}.jpg"))

# set label txt for each of the image file
# for _, entry in subset_df.iterrows():
#     img_id = entry['ImageId']
#     mask_rle = entry['EncodedPixels']
#     height = entry['Height']
#     width = entry['Width']
#     class_id = entry['ClassId']
#     utils.save_label(os.path.join(new_train_label_folder, f"{img_id}.txt"), mask_rle, (width, height), class_id)

