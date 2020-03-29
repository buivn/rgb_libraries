from __future__ import print_function
from collections import OrderedDict
import os
import torch
import torch.nn as nn
from torch import nn, optim
import torchvision.models as models
import pprint
from torchsummary import summary
from PIL import Image
import torchvision.transforms.functional as F
import time
import json
import copy

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

import torchvision.transforms.functional as TF
import random

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict


from tensorboardX import SummaryWriter
import gc
#from segnet import  SegNet
import  c_JointTransform  as joint_transforms
#from skimage import data


#image_size = 256
class PILDataset(Dataset):
    def __init__(self, imagelist,labellist, no_of_images, count, image_size, cropflag = False):
        self.transform = transforms.Compose([
            transforms.ToTensor()#,
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #self.image_files_crack = [f for f in os.listdir(trainpath) if '.JPG' in f]
        #self.label_files_crack = [f for f in os.listdir(labelpath) if '.bmp' in f]
        #self.transform = transforms.Compose([transforms.ToTensor()])
        #self.transform = transform
        #randindex = random.randint(0,no_of_images-1)
        #self.image_files_crack.sort()
        #self.label_files_crack.sort()


        #self.input_images_list, self.target_masks_list =  zip(*[joint_transforms.RandomCrop(image_size)(
        #    transforms.ToPILImage()(cv2.imread(trainpath+self.image_files_crack[i], cv2.COLOR_RGB2GRAY)),
        #    transforms.ToPILImage()(cv2.imread(labelpath+self.label_files_crack[i])))
        #    for j in range(0, count)])

        #self.input_images = self.input_images_list
        #self.target_masks = self.target_masks_list
        self.im_size = image_size
        self.input_images = []
        self.target_masks = []
        for j in range(0, count):
            if no_of_images > 1 :
                i = random.randint(0,no_of_images-1)
            else:
                i=0
            #print(i)
            image, mask = joint_transforms.RandomCrop(image_size)((imagelist[i]),(labellist[i]), cropflag)
            self.input_images.append(image)
            self.target_masks.append(mask)


        #self.input_images = self.input_images.astype('float16')
        #self.target_masks = self.target_masks.astype('float16')

        #self.input_images = torch.FloatTensor(self.input_images/ 255.0)
        #self.target_masks = torch.FloatTensor(self.target_masks

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.transform(self.input_images[idx])

        temp_mask = self.target_masks[idx]



        #print(mask.shape)
        def transform_2d(mask):
            masklabel = mask.load()
            maskfinal = numpy.zeros([self.im_size[0], self.im_size[1], 2], dtype='float32')

            for i in range(0, maskfinal.shape[1]):
                for j in range(0, maskfinal.shape[2]):
                    if masklabel[i,j] == 255:
                        maskfinal[i,j,0] = 1
                        maskfinal[i,j,1] = 0
                        #print("found")
                    else:
                        maskfinal[ i, j, 0] = 0
                        maskfinal[i, j,1] = 1
            return maskfinal

        #maskf = self.transform(transform_2d(temp_mask))#transform_2d(temp_mask)
        mask = self.transform(temp_mask)
        #print(maskf.shape)

        #image = torch.IntTensor(image)
        #mask = torch.IntTensor(mask)

        #image = Image.fromarray(image)
        #mask = Image.fromarray(mask)

        #image, mask, st = my_segmentation_transforms(image,mask)

        #print(image)
        #print(max(mask.numpy().flatten()))
        sample = {'image': image, 'mask':mask}#, 'label':mask}
        #sample = {'image': image, 'mask': mask}

        #print(sample['image'])
        #plt.imshow(numpy.transpose(sample['image'], (1,2,0)))
        return sample#[image, mask]
"""
basepath = "/media/purba/New Volume/CNN/label/UNet/"
batch_size = 20
img_size = 256
train_image_name = [f for f in os.listdir(basepath + '/train/original/') if '.JPG' in f]
label_image_name = [f for f in os.listdir(basepath + '/train/label/') if '.bmp' in f]

train_image_list = []
label_image_list = []

for i in range(0, len(train_image_name)):
    train_image_list.append(transforms.ToPILImage()(data.imread(basepath + '/train/original/' + train_image_name[i])))
    label_image_list.append(transforms.ToPILImage()(data.imread(basepath + '/train/label/' + label_image_name[i])))


train_set = PILDataset(train_image_list, label_image_list, 3, 2000, image_size=(img_size, img_size))
dataloaders = {'train': DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)}
inputs, masks, m = next(iter(dataloaders['train']))


"""