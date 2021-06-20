# -*- coding: utf-8 -*-
import argparse
import os
import random
import numpy as np
import torch
import warnings
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--mode', default='Train', choices=['Train', 'Test'], type=str, help='')
parser.add_argument('--model_name', default='efficientnet-b5', type=str, help='')
parser.add_argument('--data_local', default=r'/home/ymluo/DataSets/cassava-leaf-disease-classification', type=str, help='')
parser.add_argument('--input_size', default=384, type=int, help='600, 528, 456, 380, 300, 260, 240, 224')
parser.add_argument('--img_channel', default=3, type=int, help='')
parser.add_argument('--num_classes', default=5, type=int, help='')
parser.add_argument('--batch_size', default=64, type=int, help='')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='')
parser.add_argument('--max_epochs', default=5000, type=int, help='')

parser.add_argument('--start_epoch', default=0, type=int, help='')

parser.add_argument('--pretrained_weights', default='./models/zoo/resnext101_32x8d-8ba56ff5.pth', type=str, help='')
# parser.add_argument('--pretrained_weights', default='./models/ep00011-val_acc@1_62.5964-val_lossS_0.8536-val_lossL_0.9086.pth', type=str, help='')

# parser.add_argument('--inference_weights', default='./models/ep00016-val_acc@1_94.5051-val_lossS_0.2384-val_lossL_0.1980.pth', type=str, help='')
parser.add_argument('--inference_weights', default='./models/ep00035-val_acc@1_95.5656-val_lossS_0.2166-val_lossL_0.1691.pth', type=str, help='')

parser.add_argument('--seed', default=2021, type=int, help='0/1/2/... or None')

args, unknown = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
print('CUDA device count : {}'.format(torch.cuda.device_count()))

def check_args(args):
	print(args.data_local)
	if not os.path.exists(args.data_local):
		raise Exception('FLAGS.data_local_path: %s is not exist' % args.data_local)

def set_random_seeds(args):
	os.environ['PYTHONHASHSEED'] = str(args.seed)
	cudnn.deterministic = True
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	print('You have chosen to seed training with seed {}.'.format(args.seed))

def main(args,**kwargs):
	# check_args(args)
	if args.seed != None:
		set_random_seeds(args)
	else:
		print('You have chosen to random seed.')
	if args.mode == 'Train':
		from train import train_model
		train_model(args=args)
	elif args.mode == 'Test':
		from inference import myTest
		myTest(args)

if __name__ == '__main__':
	main(args)
