#!/usr/bin/env python3

import os
import sys
import random
import pandas as pd
from glob import glob
from tqdm import tqdm

if len(sys.argv) != 2:
    print("Usage: {} <folder>".format(sys.argv[0]))
    sys.exit(1)

folder        = sys.argv[1]
files         = glob(folder + "/*.jpg") # list all files with *.jpg extension in directory
total_size    = len(files)
indices       = list(range(total_size))

# splitting data
split_percent = 0.25
valid_n       = int(total_size * split_percent)
train_n       = total_size - valid_n
print("Total of: {} images, will use {} for training and {} for validation.".format(total_size, train_n, valid_n))

# random distribute
random.shuffle(indices)

################################
df       = pd.read_csv("../labels_combined.csv")
train_df = pd.DataFrame(columns=['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax'])
test_df  = pd.DataFrame(columns=['filename','width','height','class', 'xmin', 'ymin', 'xmax', 'ymax'])

grouped  = df.groupby('filename')

#########################################################
# Mobilenet V2
# Train
train_path = "../train"
train_csv  = "../train_labels.csv"
if not os.path.exists(train_path):
    os.mkdir(train_path)

print("Train Images Copying")
for i in tqdm(indices[:train_n]):
    # images
    cp_img = "cp {} {}/".format(files[i], train_path)
    os.system(cp_img)
    
    # labels
    img_name = files[i].split('/')[2]
    info_df  = grouped.get_group(img_name)
    train_df = pd.concat([train_df, info_df])     

train_df = train_df.reset_index()
train_df = train_df.drop(['index'], axis=1)
train_df.to_csv(train_csv, index=False)

# Test
test_path = "../test"
test_csv  = "../test_labels.csv"
if not os.path.exists(test_path):
    os.mkdir(test_path)

print("Test Images Copying")
for i in tqdm(indices[train_n:]):
    # images
    cp_img = "cp {} {}/".format(files[i], test_path)
    os.system(cp_img)

    # labels
    img_name = files[i].split('/')[2]
    info_df  = grouped.get_group(img_name)
    test_df  = pd.concat([test_df, info_df])     

test_df = test_df.reset_index()
test_df = test_df.drop(['index'], axis=1)
test_df.to_csv(test_csv, index=False)