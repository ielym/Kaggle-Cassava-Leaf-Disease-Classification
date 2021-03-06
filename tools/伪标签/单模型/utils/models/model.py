import torchvision
import torch
import torch.nn as nn
import collections
import cv2
import numpy as np
from .resnext101_32x8d import resnext101_32x8d

def _resnet101(weights, num_classes):
	model = resnext101_32x8d(pretrained=False, num_classes=num_classes)

	if weights:
		pretrained_dict = torch.load(weights)
		# pretrained_dict.pop('fc')
		model.load_state_dict(pretrained_dict, strict=False)
	return model

def ResNet101(weights, input_shape, num_classes):
	model = _resnet101(weights, num_classes)
	return model

if __name__ == '__main__':
	from torchsummary import summary
	model = ResNet101(weights=None, input_shape=(3, 224, 224), num_classes=3)
	summary(model, (3, 224, 224), device='cpu')

# import tensorwatch as tw
	# img = torch.ones(size=(1, 3, 224, 224))
	# tw.draw_model(model, (1, 3, 224, 224), png_filename='./res.png')
