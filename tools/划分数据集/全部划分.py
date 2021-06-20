import numpy as np
import pandas as pd
import os
from glob import glob

import cv2

base_dir = r'S:\DataSets\cassava-leaf-disease-classification'

label_path = os.path.join(base_dir, 'train.csv')
img_dir = os.path.join(base_dir, 'train_images')

test_size = 0.1

df = pd.read_csv(label_path)
image_id_label = df.to_numpy()
np.random.shuffle(image_id_label)

total_imgs = len(image_id_label)
print('Total samples : {}'.format(total_imgs))

train_id_labels = image_id_label[int(test_size * total_imgs) :]
val_id_labels = image_id_label[0 : int(test_size * total_imgs)]

train_id_labels = np.array(train_id_labels).reshape([-1, 2])
val_id_labels = np.array(val_id_labels).reshape([-1, 2])

print("Train samples : {}".format(train_id_labels.shape))
print("Validation samples : {}".format(val_id_labels.shape))

train_df = pd.DataFrame(train_id_labels, index=None, columns=['image_id', 'label'])
train_df.to_csv(os.path.join(base_dir, 'train_split.csv'), index=False)

val_df = pd.DataFrame(val_id_labels, index=None, columns=['image_id', 'label'])
val_df.to_csv(os.path.join(base_dir, 'val_split.csv'), index=False)
