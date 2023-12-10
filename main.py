import os
import pandas as pd
from collections import Counter

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


