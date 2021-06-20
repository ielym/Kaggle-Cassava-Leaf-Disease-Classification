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
		RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
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
model1 = torch.jit.load(r'./models/src33-4-ep00034-val_acc@1_89.2290-val_lossFocalCosine_0.4061.tjm')
model2 = torch.jit.load(r'./models/src33-1-ep00023-val_acc@1_89.2523-val_lossFocalCosine_0.4197.tjm')
model3 = torch.jit.load(r'./models/src33-2-ep00022-val_acc@1_88.5514-val_lossFocalCosine_0.4115.tjm')
model4 = torch.jit.load(r'./models/src33-0-ep00027-val_acc@1_89.3692-val_lossFocalCosine_0.4040.tjm')
model5 = torch.jit.load(r'./models/src33-3-ep00024-val_acc@1_89.4159-val_lossFocalCosine_0.4031.tjm')
model6 = torch.jit.load(r'./models/src28-ep00020-val_acc@1_88.tjm')

model1 = model1.cuda()
model2 = model2.cuda()
model3 = model3.cuda()
model4 = model4.cuda()
model5 = model5.cuda()
model6 = model6.cuda()
model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()

Target_DIR = r'S:\DataSets\cassavaVersion\V4'
TEST_DIR = r'S:\DataSets\cassava-disease-2019-test'
test_images = os.listdir(TEST_DIR)

img_size = (384, 384)

predictions = []
prediction_images = []

with torch.no_grad():
	count = 0
	for image_name in test_images:
		img = preprocess_img(os.path.join(TEST_DIR, image_name))
		img = img.cuda()

		out1, _ = model1(img)
		scores1 = torch.softmax(out1, dim=1).cpu().numpy()[0]
		out1 = out1.cpu().numpy()
		categories1 = np.argmax(out1, axis=1)

		out2, _ = model2(img)
		scores2 = torch.softmax(out2, dim=1).cpu().numpy()[0]
		out2 = out2.cpu().numpy()
		categories2 = np.argmax(out2, axis=1)

		out3, _ = model3(img)
		scores3 = torch.softmax(out3, dim=1).cpu().numpy()[0]
		out3 = out3.cpu().numpy()
		categories3 = np.argmax(out3, axis=1)

		out4, _ = model4(img)
		scores4 = torch.softmax(out4, dim=1).cpu().numpy()[0]
		out4 = out4.cpu().numpy()
		categories4 = np.argmax(out4, axis=1)

		out5, _ = model5(img)
		scores5 = torch.softmax(out5, dim=1).cpu().numpy()[0]
		out5 = out5.cpu().numpy()
		categories5 = np.argmax(out5, axis=1)

		out6, _ = model6(img)
		scores6 = torch.softmax(out6, dim=1).cpu().numpy()[0]
		out6 = out6.cpu().numpy()
		categories6 = np.argmax(out6, axis=1)

		categories = np.hstack([categories1, categories2, categories3, categories4, categories5, categories6])
		counts = np.bincount(categories)
		category = np.argmax(counts)

		score1 = scores1[category]
		score2 = scores2[category]
		score3 = scores3[category]
		score4 = scores4[category]
		score5 = scores5[category]
		score6 = scores6[category]

		score = np.mean([score1, score2, score3, score4, score5, score6])

		count += 1
		print('\r{}'.format(count), end='')

		if score >= 0.90:
			# ori_img = cv2.imread(os.path.join(TEST_DIR, image_name))
			# ori_img = cv2.resize(ori_img, (img_size))
			# cv2.imshow('{} - {} - {}'.format(image_name, category, score), ori_img)
			# cv2.waitKey()
			# shutil.move(os.path.join(TEST_DIR, image_name), Target_DIR)
			# shutil.copy(os.path.join(TEST_DIR, image_name), Target_DIR)
			predictions.append(category)
			prediction_images.append(image_name)

sub = pd.DataFrame({'image_id': prediction_images, 'label': predictions})
sub.to_csv(os.path.join(Target_DIR, 'train.csv'), index=False)