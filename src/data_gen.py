# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import math
import codecs
import random
import numpy as np
from glob import glob
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from torchtoolbox.transform import Cutout
from PIL import Image
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

class DataAugmentation():

	def Train_Transforms(self, img_size):
		return Compose([
			RandomResizedCrop(img_size[0], img_size[1]),
			Transpose(p=0.5),
			HorizontalFlip(p=0.5),
			VerticalFlip(p=0.5),
			ShiftScaleRotate(p=0.5),
			HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
			RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
			CoarseDropout(p=0.5),
			Cutout(p=0.5, max_h_size=20, max_w_size=20),
			ToTensorV2(p=1.0),
		], p=1.)

	def Val_Transforms(self, img_size):
		return Compose([
			Resize(img_size[0], img_size[1]),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
			ToTensorV2(p=1.0),
		], p=1.)

	def Test_Transforms(self):
		pass

class BaseDataset(Dataset):
	def __init__(self, paths_labels, input_shape, mode, num_classes):
		self.samples = paths_labels
		self.input_shape = input_shape
		self.mode = mode
		self.num_classes = num_classes

		self.augmentation = DataAugmentation()
		self.train_transforms = self.augmentation.Train_Transforms(img_size=(self.input_shape[1], self.input_shape[2]))
		self.val_transforms = self.augmentation.Val_Transforms(img_size=(self.input_shape[1], self.input_shape[2]))

	def __len__(self):
		return len(self.samples)

	def preprocess_img(self, img_path):
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		if self.mode == 'train':
			img = self.train_transforms(image=img)['image']
		else:
			img = self.val_transforms(image=img)['image']

		return img

	def preprocess_label(self, category):
		label = np.zeros(shape=[self.num_classes])
		label[category] = 1
		label.astype(np.float32)
		return label

	def __getitem__(self, idx):
		x = self.preprocess_img(self.samples[idx, 0])
		y = self.preprocess_label(self.samples[idx, 1])
		y = torch.tensor(y, dtype=torch.float32)
		# y = torch.tensor(y, dtype=torch.float32).view(1, -1)
		return x, y

def data_flow(base_dir, input_shape, num_classes):

	all_label_path = os.path.join(base_dir, 'train.csv')
	images_dir = os.path.join(base_dir, 'train_images')

	all_df = pd.read_csv(all_label_path)
	all_path_labels = all_df.to_numpy()

	for line in all_path_labels:
		line[0] = os.path.join(images_dir, line[0])

	all_paths = all_path_labels[:, 0]
	all_labels = all_path_labels[:, 1]

	k = 0
	K_train_indexs = []
	K_test_indexs = []
	ss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=2021)
	for train_index, test_index in ss.split(all_paths, all_labels):
		K_train_indexs.append(train_index)
		K_test_indexs.append(test_index)
	train_paths, test_paths = all_paths[K_train_indexs[k]], all_paths[K_test_indexs[k]]
	train_labels, test_labels = all_labels[K_train_indexs[k]], all_labels[K_test_indexs[k]]

	train_name_category = np.hstack([train_paths.reshape([-1, 1]), train_labels.reshape([-1, 1])])
	val_name_category = np.hstack([test_paths.reshape([-1, 1]), test_labels.reshape([-1, 1])])

	print('total samples: %d, training samples: %d, validation samples: %d' % (len(train_name_category) + len(val_name_category), len(train_name_category), len(val_name_category)))
	train_dataset = BaseDataset(train_name_category, input_shape, 'train', num_classes)
	validation_dataset = BaseDataset(val_name_category, input_shape, 'val', num_classes)

	return train_dataset, validation_dataset


if __name__ == '__main__':

	# data_flow(r'S:\DataSets\cassava-leaf-disease-classification', input_shape=(3, 500, 500), num_classes=5)
	train_dataset, validation_dataset = data_flow(r'S:\DataSets\cassava-leaf-disease-classification', input_shape=(3, 500, 500), num_classes=5)
	epoch = 1
	while True:
		epoch += 1
		for i in validation_dataset:
			img = i[0].numpy()
			label = i[1].numpy()
			img = np.transpose(img, [1, 2, 0])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			print(label, img.shape)
			cv2.imshow(str(np.argmax(label, axis=1)), img)
			cv2.waitKey()
			cv2.destroyAllWindows()
