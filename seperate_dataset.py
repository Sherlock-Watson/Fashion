import utils

import pandas as pd
import json

with open("paths.json") as file:
    paths = json.load(file)

new_csv_path = paths['new_csv_path']
whole_df = pd.read_csv(new_csv_path)
df = whole_df[['ImageId', 'ClassId']]

# threshold = 40
class_list = list(set(df['ClassId']))

first_file_set = set()
second_file_set = set()
for class_id in class_list:
    sub_file_list = df[df['ClassId'] == class_id]['ImageId']
    threshold = len(sub_file_list) / 15
    # sub_file_list = random.sample(sub_file_list, len(sub_file_list))
    first_sub_file_set = set()
    count = 0
    amount = threshold - df[(df['ClassId'] == class_id) & (df['ImageId'].isin(first_file_set))].shape[0]
    stop_index = 0
    # fetch the samples for the test folder
    for (index, file) in enumerate(sub_file_list):
        stop_index = index
        if (file in first_file_set) or (file in second_file_set):
            continue
        count += 1
        first_sub_file_set.add(file)
        if count >= amount:
            break
    first_file_set = first_file_set.union(first_sub_file_set)
    second_sub_file_set = set()
    count = 0
    amount = threshold - df[(df['ClassId'] == class_id) & (df['ImageId'].isin(second_file_set))].shape[0]
    # According to the sequence, fetch the next folder
    for index in range(stop_index + 1, len(sub_file_list)):
        file = sub_file_list.iloc[index]
        if (file in first_file_set) or (file in second_file_set):
            continue
        count += 1
        second_sub_file_set.add(file)
        if count >= amount:
            break
    second_file_set = second_file_set.union(second_sub_file_set)
assert len(first_file_set & second_file_set) == 0, "The test and validation set intersects"
print(f"Image count for test folder: {len(first_file_set)}")
utils.print_class_id_distribution(df[df['ImageId'].isin(first_file_set)], "ClassId")
print(f"Image count for validation folder: {len(second_file_set)}")
utils.print_class_id_distribution(df[df['ImageId'].isin(second_file_set)], "ClassId")
total_file_set = set(df['ImageId'])
train_file_set = total_file_set - first_file_set - second_file_set
assert len(first_file_set & train_file_set) == 0, "The test and train set intersects"
assert len(train_file_set & second_file_set) == 0, "The train and validation set intersects"
assert len(train_file_set) + len(first_file_set) + len(second_file_set) == len(total_file_set), \
    "The three subsets can't reconstruct the whole set"
print(f"Image count for train folder: {len(train_file_set)}")
utils.print_class_id_distribution(df[df['ImageId'].isin(train_file_set)], 'ClassId')
train_df = whole_df[whole_df['ImageId'].isin(train_file_set)]
train_df.to_csv(paths['new_train_csv'])
test_df = whole_df[whole_df['ImageId'].isin(first_file_set)]
test_df.to_csv(paths['new_test_csv'])
val_df = whole_df[whole_df['ImageId'].isin(second_file_set)]
val_df.to_csv(paths['new_val_csv'])
