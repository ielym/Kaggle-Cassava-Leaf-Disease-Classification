# -*- coding: utf-8 -*-
import multiprocessing
import time
import torch
import torch.nn as nn
from torchsummary import summary
import torch_optimizer as optim  # https://github.com/jettify/pytorch-optimizer

from cfg import _metrics, _fit, _modelcheckpoint, _reducelr, _criterion
# from gpu_selection import singleGPU, multiGPU, SingleModelCheckPoint, ParallelModelCheckpoint
# from utils import *
from data_gen import data_flow
from models.model import ResNet101, Efficient

import os

def model_fn(args):
	model = ResNet101(weights=args.pretrained_weights, input_shape=(args.img_channel, args.input_size, args.input_size), num_classes=args.num_classes)
	# model = ResNet101(weights=None, input_shape=(args.img_channel, args.input_size, args.input_size), num_classes=args.num_classes)
	# pretrained_dict = torch.load(args.pretrained_weights)
	# single_dict = {}
	# for k, v in pretrained_dict.items():
	# 	single_dict[k[7:]] = v
	# model.load_state_dict(single_dict, strict=True)

	for param in model.parameters():
		param.requires_grad = True

	for name, value in model.named_parameters():
		if 'fc' in name:
			value.requires_grad = True
		if 'lym' in name:
			value.requires_grad = True
		# if not name.startswith('layer'):
		# 	value.requires_grad = True

	for name, value in model.named_parameters():
		print(name, value.requires_grad)

	model = nn.DataParallel(model)
	model = model.cuda()

	return model

def train_model(args):
	model = model_fn(args)

	# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	radam = optim.Ranger(params=model.parameters(), lr=args.learning_rate)
	optimizer = optim.Lookahead(radam)
	# optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)

	criterion = {'lossFocalCosine' : _criterion.FocalCosineLoss().cuda(), 'lossBTL' : _criterion.Bi_Tempered_Logistic_Loss(label_smoothing=0.1, t1=0.8, t2=1.4, num_iters=5, reduction='mean').cuda()}
	# criterion = {'lossCE' : nn.CrossEntropyLoss().cuda(), 'lossLSCE' : _criterion.LabelSmoothSoftmaxCE().cuda()}
	# criterion = {'lossBTL' : _criterion.Weighted_Bi_Tempered_Logistic_Loss(balance=[12.1,6.0,5.5,1.0,5.1], label_smoothing=0.1, t1=0.8, t2=1.4, num_iters=5, reduction='mean').cuda()}
	# criterion = {'lossL' : _criterion.FocalLoss(class_num=3, alpha=None, gamma=2, size_average=True).cuda(), 'lossS' : _criterion.LabelSmoothSoftmaxCE().cuda()}
	metrics = {"acc@1" : _metrics.top1_accuracy, "acc@3" : _metrics.topk_accuracy}

	checkpoint1 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_acc@1_{val_acc@1:.4f}-val_lossFocalCosine_{val_lossFocalCosine:.4f}.pth'), monitor='val_acc@1', mode='max', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint2 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}-val_acc@1_{val_acc@1:.4f}-val_lossFocalCosine_{val_lossFocalCosine:.4f}.pth'), monitor='val_lossFocalCosine', mode='min', verbose=1, save_best_only=True, save_weights_only=True)
	checkpoint3 = _modelcheckpoint.SingleModelCheckPoint(filepath=os.path.join('./models/', 'ep{epoch:05d}.pth'), monitor='val_acc@1', mode='min', verbose=1, save_best_only=False, save_weights_only=True)

	# reduce_lr = _reducelr.StepLR(optimizer, factor=0.2, patience=10, min_lr=1e-6)
	reduce_lr = _reducelr.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-6)

	_fit.Fit(
			data_flow = data_flow,
			model=model,
			args=args,
			batch_size = args.batch_size,
			optimizer=optimizer,
			criterion=criterion,
			metrics=metrics,
			reduce_lr = reduce_lr,
			checkpoint = [checkpoint1, checkpoint2, checkpoint3],
			verbose=1,
			workers=int(multiprocessing.cpu_count() * 0.8),
		)
	print('training done!')


