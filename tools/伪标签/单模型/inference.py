import torch
import torchvision
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
from glob import glob
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from albumentations import (
	HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
	Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
	IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
	IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
	ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

def tta_ori(img_size):
	return Compose([
		Resize(img_size[0], img_size[1]),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
		ToTensorV2(p=1.0),
	], p=1.)

def tta_hflip(img_size):
	return Compose([
		Resize(img_size[0], img_size[1]),
		HorizontalFlip(p=1.0),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
		ToTensorV2(p=1.0),
	], p=1.)


def tta_vflip(img_size):
	return Compose([
		Resize(img_size[0], img_size[1]),
		VerticalFlip(p=1.0),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
		ToTensorV2(p=1.0),
	], p=1.)


def tta_transpose(img_size):
	return Compose([
		Resize(img_size[0], img_size[1]),
		Transpose(p=1.0),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
		ToTensorV2(p=1.0),
	], p=1.)


def tta_BC(img_size):
	return Compose([
		Resize(img_size[0], img_size[1]),
		RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=1.0),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
		ToTensorV2(p=1.0),
	], p=1.)


def tta_CenterCrop(img_size):
	return Compose([
		Resize(img_size[0] + 64, img_size[1] + 64),
		CenterCrop(img_size[0], img_size[1], p=1.0),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
		ToTensorV2(p=1.0),
	], p=1.)


def tta_RandomResizeCrop(img_size):
	return Compose([
		RandomResizedCrop(img_size[0], img_size[1]),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
		ToTensorV2(p=1.0),
	], p=1.)


def preprocess_img(img_path):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	img_tta1 = tta_ori(img_size=(384, 384))(image=img)['image']
	img_tta2 = tta_hflip(img_size=(384, 384))(image=img)['image']
	img_tta3 = tta_vflip(img_size=(384, 384))(image=img)['image']
	img_tta4 = tta_transpose(img_size=(384, 384))(image=img)['image']
	img_tta5 = tta_BC(img_size=(384, 384))(image=img)['image']
	img_tta6 = tta_CenterCrop(img_size=(384, 384))(image=img)['image']
	img_tta7 = tta_RandomResizeCrop(img_size=(384, 384))(image=img)['image']

	cat_img = []
	cat_img.append(img_tta1)
	cat_img.append(img_tta2)
	cat_img.append(img_tta3)
	cat_img.append(img_tta4)
	cat_img.append(img_tta5)
	cat_img.append(img_tta6)
	cat_img.append(img_tta7)

	img = torch.stack(cat_img)

	return img


# model = model_fn(r'./models/src28-ep00020-val_acc@1_88.tjm')
model = torch.jit.load(r'./models/src28-ep00020-val_acc@1_88.tjm')
model = model.cuda()

Target_DIR = r'S:\DataSets\cassavaVersion\V1'
TEST_DIR = r'S:\DataSets\cassava-disease-2019-train\train\cmd'
test_images = os.listdir(TEST_DIR)

img_size = (384, 384)

predictions = []
prediction_images = []
model.eval()
with torch.no_grad():
	count = 0
	for image_name in test_images:
		img = preprocess_img(os.path.join(TEST_DIR, image_name))
		img = img.cuda()
		out, _ = model(img)
		scores = torch.softmax(out, dim=1).cpu().numpy()[0]
		out = out.cpu().numpy()
		categories = np.argmax(out, axis=1)
		counts = np.bincount(categories)
		category = np.argmax(counts)
		score = scores[category]

		if score >= 0.90 and category != 3:
			count += 1
			print('\r{}'.format(count), end='')
			print(image_name)
			# ori_img = cv2.imread(os.path.join(TEST_DIR, image_name))
			# ori_img = cv2.resize(ori_img, (img_size))
			# cv2.imshow('{} - {} - {}'.format(image_name, category, score), ori_img)
			# cv2.waitKey()
			predictions.append(category)
			prediction_images.append(image_name)

# sub = pd.DataFrame({'image_id': prediction_images, 'label': predictions})
# sub.to_csv(os.path.join(Target_DIR, 'train.csv'), index=False)