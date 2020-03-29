#!/home/buivn/bui_virenv/crackDetection/bin/python3
import  torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils import model_zoo
from torchvision import models
import torchvision.transforms as transforms
from PIL import ImageOps as op
import time
import  os
import numpy
from skimage import data, io
import imageio as io
from PIL import Image
from openpyxl import load_workbook
import pandas as pd

from c_PILDataset import  PILDataset
from c_utils import  *
from c_SegnetHED import *
from c_unet import *



class crack_detection:
  """docstring for crack_detection"""
  def __init__(self, path, modelAddress):
    self.basepath = path
    self.foldername = ""
    self.segnet = SegNetHEDKernel7(num_classes=1, filter_config=(4, 8, 16, 32, 64), kernel_size=7, padding=3).cuda()
    self.modelAddress = modelAddress
    # self.resultAddress = 


  def detect_crack(self):
    transform1 = transforms.Compose([transforms.ToPILImage()])

    self.segnet.load_state_dict(torch.load(self.modelAddress))
    tf = transforms.Compose([transforms.ToTensor()])
    
    model = self.segnet
    model.eval()
    # process on one image
    for i in range(0, 1):
      image = data.imread(self.basepath+"/inputs/onground4.jpg")
      Label = numpy.zeros([image.shape[0], image.shape[1], 3], dtype=numpy.float)
      ImageOrig = numpy.zeros([image.shape[0], image.shape[1], 3], dtype=numpy.float)
      start_pos = 0
      image_size = [image.shape[0], image.shape[1], image.shape[2]]
      # temp_image = []
      # temp_coordinate = []
      x = image.shape[0]
      y = image.shape[1]
      sub_image_x = int(x / 800)
      sub_image_y = int(y / 800)

      subimage_size = 800
      start = start_pos
      end = 800 + start_pos

      for j in range(0, sub_image_x ):
        start1 = start_pos
        end1 = 800 + start_pos
        for k in range(0, sub_image_y ):
          size_x = 800
          size_y = 800
          if end > image_size[0]:
            size_x = image_size[0]-start
            end = image_size[0]
          if end1 > image_size[1]:
            size_y = image_size[1] - start1
            end1 = image_size[1]

          img =image[start:end, start1:end1][:] 

          io.imsave(self.basepath+"/intermediate/image_pos_"+str(start)+"_"+str(start1)+".png", img)
          valid_image_list = []
          valid_label_list = []
          valid_image_list.append(transform1(img))
          valid_label_list.append(transform1(img))
          # coordinate = (start, start1)
      
          detect_set = PILDataset(valid_image_list, valid_label_list, len(valid_label_list), 1, image_size=(size_x, size_y), cropflag=True)
          dataloaders = {'detect': DataLoader(detect_set, batch_size=1, shuffle=False, num_workers=0)}
          
          for k, data in enumerate(dataloaders['train']):
            torch.cuda.empty_cache()
            x = data['image']
            # x = transform(x)
            X = Variable(x).cuda()  
            Y = data['mask']
            Y = Variable(Y).cuda()
            output = model(X)
            pre = output[0].cpu()
            pred = pre.detach()
            out = transform(pred)
            x_numpy = numpy.array(transform(x[0]))
            out_numpy = pred.numpy()
            out_numpy = out_numpy[0]  # numpy.transpose(out_numpy, (2,1,0))
            threshold = Threshold(out_numpy)

            th_img = threshold.threshold(0.70, skip=0)
            Label[start:end, start1:end1] = th_img
            ImageOrig[start:end, start1:end1] = x_numpy
          start1 += 800
          end1 += 800
        start += 800
        end += 800

      # io.imsave("/home/buivn/Desktop/ANet/stitched/Predicted.png", Label)
      io.imsave(self.basepath+"/results/Predicted.png", Label)
      io.imsave(self.basepath+"/results/Original.png", ImageOrig)


"""
for i in range(0, len(valid_image_name)):
    image = Image.open(basepath + '//original/' + valid_image_name[i])
    valid_image_list.append(image)
"""
