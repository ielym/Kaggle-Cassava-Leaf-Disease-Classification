import os
import cv2
import numpy as np
import torch
from torchvision.datasets import ImageFolder

basr_dir = r'/home/ymluo/DataSets/cassava-leaf-disease-classification/train_images'
files = os.listdir(basr_dir)
print('1321231312')
# one = []
# two = []
# three = []
# for i in files:
#     img = cv2.imread(os.path.join(basr_dir,i))
#     one.append((img[:,:,0]/255).mean())
#     two.append((img[:,:,1]/255).mean())
#     three.append((img[:,:,2]/255).mean())
#
# print(sum(one)/len(files))
# print(sum(two)/len(files))
# print(sum(three)/len(files))

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



num = len(files) * 800 * 600

imgs = []
R_channel = 0
G_channel = 0
B_channel = 0
for i in range(0,len(files)):
    img = cv2.imread(os.path.join(basr_dir, files[i]))
    R_channel = R_channel + np.sum(img[:, :, 0]/255)
    G_channel = G_channel + np.sum(img[:, :, 1]/255)
    B_channel = B_channel + np.sum(img[:, :, 2]/255)

R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
j = 0
for i in range(0,len(files)):
    print(j)
    img = cv2.imread(os.path.join(basr_dir, files[i]))
    R_channel = R_channel + np.sum(np.power(img[:, :, 0]/255 - R_mean, 2))
    G_channel = G_channel + np.sum(np.power(img[:, :, 1]/255 - G_mean, 2))
    B_channel = B_channel + np.sum(np.power(img[:, :, 2]/255- B_mean, 2))
    j += 1

R_std = np.sqrt(R_channel / num)
G_std = np.sqrt(G_channel / num)
B_std = np.sqrt(B_channel / num)

# R:65.045966   G:70.3931815    B:78.0636285
print("B_mean is %f, G_mean is %f, R_mean is %f" % (R_mean, G_mean, B_mean))
print("B_std is %f, G_std is %f, R_std is %f" % (R_std, G_std, B_std))