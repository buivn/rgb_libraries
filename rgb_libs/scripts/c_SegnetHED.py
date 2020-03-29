import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import  torch
import time
import  os
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy
import torchvision.transforms as transforms
from c_PILDataset import  PILDataset
#from skimage import  data,io
#from EncoderDecoder import SegNet1
from PIL import Image
from openpyxl import load_workbook
import pandas as pd

from c_utils import Superimpose, PILImageUtility, PILImageSave, Threshold

def conv1x1(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)
def conv5x5(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=5,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)
def conv7x7(in_channels, out_channels, stride=1, padding=0, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=7,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)
def conv11x11(in_channels, out_channels, stride=1, padding=0, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=11,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)
def Upconv(in_channels, out_channels, stride=1, padding=3, kernel_size = 7):
    if kernel_size == 7:
        return nn.Sequential(conv7x7(in_channels, out_channels, stride, padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))
    elif kernel_size==5:
        return nn.Sequential(conv5x5(in_channels, out_channels, stride=1, padding=2),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True))
    elif kernel_size == 3:
        return nn.Sequential(conv3x3(in_channels, out_channels, stride=1, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))
def side_output( filter, kernel, factor):

   return nn.Sequential(
    #nn.Conv2d(filter,filter, kernel_size=kernel, stride=1, padding=3),
    nn.UpsamplingBilinear2d(scale_factor=factor))
    #nn.UpsamplingBilinear2d(scale_factor=factor))
    #nn.ConvTranspose2d(filter, filter, kernel_size=factor, stride=factor, padding=0))


class SegNetHEDKernel7(nn.Module):

    def __init__(self, num_classes=1, n_init_features=3, drop_rate=0.5,
                 filter_config=(4, 8, 16, 32, 64), kernel_size = 7, padding = 3):
        super(SegNetHEDKernel7, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.sides = nn.ModuleList()
        self.filter_config = filter_config

        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 2)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)
        factor = [2,4,8,16,32]
        kernel = [4,8,16,32,32]
        #print(encoder_filter_config)

        for i in range(0, len(self.filter_config)):

            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          1, kernelsize=kernel_size))

            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i +1],
                                          1, kernelsize=kernel_size))

            self.sides.append(Sideoutput(encoder_filter_config[i+1], kernel=kernel_size,factor=factor[i]))


        self.classifier = nn.Sequential(nn.Conv2d(filter_config[0], num_classes, kernel_size, 1, 1),
                         nn.BatchNorm2d(num_classes),
                         #nn.Sigmoid())
                         nn.ReLU(inplace=True))

        #final_filter = sum(filter_config)+filter_config[0]+6+12
        final_filter = sum(filter_config)+filter_config[0]
        #print(final_filter)
        self.classifier1 = nn.Sequential(nn.Conv2d(final_filter, num_classes, kernel_size, 1, padding),
                         nn.BatchNorm2d(num_classes),
                         #nn.Sigmoid())
                         nn.ReLU(inplace=True))
    def forward(self, x):
        indice = []
        unpool_sizes = []
        feat = x
        ind = None

        side = []
        factor = 2
        pruned_side = []
        for i in range(0, len(self.filter_config)):
            #print(self.encoders[i])
            feat,ind = self.encoders[i](feat)

            indice.append(ind)
            #print(feat.size())
            b1 = torch.nn.Sigmoid()(self.sides[i](feat))#self.side_conv1_1(feat)
            side.append(b1)

        for i in range(0, len(self.filter_config)):

            unpool, feat = self.decoders[i](feat, indice[len(self.filter_config)-1-i])

        xx = torch.cat([side[0], side[1], side[2], side[3], side[4], feat], dim=1)

        #xx = torch.cat([ pruned_side[0], pruned_side[1],pruned_side[2], pruned_side[3],feat], dim = 1)
        #print(xx.size())
        xx = self.classifier1(xx)

        final = (xx)

        #print(final.size())
        return F.softmax(final, dim=2)


class SegNetHEDKernel7Pruned(nn.Module):

    def __init__(self, num_classes=3, n_init_features=3, drop_rate=0.5,
                 filter_config=(4, 8, 16, 32, 64), kernel_size = 7, padding = 3):
        super(SegNetHEDKernel7Pruned, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.sides = nn.ModuleList()
        self.filter_config = filter_config

        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 2)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)
        factor = [2,4,8,16,32]
        kernel = [4,8,16,32,32]
        #print(encoder_filter_config)

        for i in range(0, len(self.filter_config)):

            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          1, kernelsize=kernel_size))

            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i +1],
                                          1, kernelsize=kernel_size))

            self.sides.append(Sideoutput(encoder_filter_config[i+1], kernel=kernel_size,factor=factor[i]))


        self.classifier = nn.Sequential(nn.Conv2d(filter_config[0], num_classes, kernel_size, 1, 1),
                         nn.BatchNorm2d(num_classes),
                         #nn.Sigmoid())
                         nn.ReLU(inplace=True))

        #final_filter = sum(filter_config)+filter_config[0]+6+12
        final_filter =  filter_config[0]+61
        #print(final_filter)
        self.classifier1 = nn.Sequential(nn.Conv2d(final_filter, num_classes, kernel_size, 1, padding),
                         nn.BatchNorm2d(num_classes),
                         #nn.Sigmoid())
                         nn.ReLU(inplace=True))
    def forward(self, x):
        indice = []
        unpool_sizes = []
        feat = x
        ind = None

        side = []
        factor = 2
        pruned_side = []
        for i in range(0, len(self.filter_config)):
            #print(self.encoders[i])
            feat,ind = self.encoders[i](feat)

            indice.append(ind)
            #print(feat.size())
            b1 = torch.nn.Sigmoid()(self.sides[i](feat))#self.side_conv1_1(feat)

            if i ==1 :
                #print(feat.size())
                enc = torch.cat([feat[:, 1:3, :,:], feat[:, 4:5, :,:], feat[:, 7:8, :,:]], dim = 1)
                enc  = torch.nn.Sigmoid()(Sideoutput(2, kernel=7, factor=4)(enc))
                pruned_side.append(enc)
            elif i ==2:
                enc = torch.cat([feat[:, 0:2, :, :], feat[:, 3:5, :, :], feat[:, 6:7, :, :], feat[:, 8:9, :, :],feat[:, 11:12, :,:]], dim=1)

                enc = torch.nn.Sigmoid()(Sideoutput(2, kernel=7, factor=8)(enc))
                pruned_side.append(enc)
            elif i ==3:
                enc = torch.cat([feat[:, 1:2, :,:],
                                feat[:, 4:6, :, :],
                                 feat[:, 7:13, :, :],
                                 feat[:, 18:19, :, :],
                                 feat[:, 20:23, :, :],
                                 feat[:, 24:25, :, :],
                                 feat[:, 26:27, :,:],
                                 feat[:, 29:30, :, :]],
                                dim=1)
                enc = torch.nn.Sigmoid()(Sideoutput(2, kernel=7, factor=16)(enc))
                pruned_side.append(enc)
            elif i == 4:
                enc = torch.cat([feat[:, 0:1, :, :],
                                feat[:, 4:5, :, :],
                                 feat[:, 7:8, :, :],
                                 feat[:, 10:12, :, :],
                                 feat[:, 14:15, :, :],
                                 feat[:, 16:17, :, :],
                                 feat[:, 18:19, :, :],
                                 feat[:, 24:26, :, :],
                                 feat[:, 27:33, :, :],
                                 feat[:, 37:43, :, :],
                                 feat[:, 44:45, :, :],
                                 feat[:, 46:49, :, :],
                                 feat[:, 51:53, :, :],
                                 feat[:, 54:56, :, :],
                                 feat[:, 59:60, :, :],
                                 feat[:, 61:64, :, :]],dim=1)
                enc = torch.nn.Sigmoid()(Sideoutput(2, kernel=7, factor=32)(enc))
                pruned_side.append(enc)
               # print(enc.size())

            side.append(b1)

        for i in range(0, len(self.filter_config)):

            unpool, feat = self.decoders[i](feat, indice[len(self.filter_config)-1-i])


        #xx = torch.cat([side[0], side[1], side[2], side[3], side[4], pruned_side[0], pruned_side[1],pruned_side[2],feat],dim=1)

        xx = torch.cat([ pruned_side[0], pruned_side[1],pruned_side[2], pruned_side[3],feat], dim = 1)
        #print(xx.size())
        xx = self.classifier1(xx)

        final = (xx)

        #print(final.size())
        return F.softmax(final, dim=2)


class SegNetHEDKernel7PrunedSpalling(nn.Module):

    def __init__(self, num_classes=3, n_init_features=3, drop_rate=0.5,
                 filter_config=(4, 8, 16, 32, 64), kernel_size = 7, padding = 3):
        super(SegNetHEDKernel7PrunedSpalling, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.sides = nn.ModuleList()
        self.filter_config = filter_config

        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 2)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)
        factor = [2,4,8,16,32]
        kernel = [4,8,16,32,32]
        #print(encoder_filter_config)

        for i in range(0, len(self.filter_config)):

            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          1, kernelsize=kernel_size))

            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i +1],
                                          1, kernelsize=kernel_size))

            self.sides.append(Sideoutput(encoder_filter_config[i+1], kernel=kernel_size,factor=factor[i]))


        self.classifier = nn.Sequential(nn.Conv2d(filter_config[0], num_classes, kernel_size, 1, 1),
                         nn.BatchNorm2d(num_classes),
                         #nn.Sigmoid())
                         nn.ReLU(inplace=True))

        #final_filter = sum(filter_config)+filter_config[0]+6+12
        final_filter =  filter_config[0]+5
        #print(final_filter)
        self.classifier1 = nn.Sequential(nn.Conv2d(final_filter, num_classes, kernel_size, 1, padding),
                         nn.BatchNorm2d(num_classes),
                         #nn.Sigmoid())
                         nn.ReLU(inplace=True))
    def forward(self, x):
        indice = []
        unpool_sizes = []
        feat = x
        ind = None

        side = []
        factor = 2
        pruned_side = []
        for i in range(0, len(self.filter_config)):
            #print(self.encoders[i])
            feat,ind = self.encoders[i](feat)

            indice.append(ind)
            #print(feat.size())
            b1 = torch.nn.Sigmoid()(self.sides[i](feat))#self.side_conv1_1(feat)

            if i ==1 :
                #print(feat.size())
                enc = torch.cat([feat[:, 1:3, :,:], feat[:, 4:5, :,:], feat[:, 7:8, :,:]], dim = 1)
                enc  = torch.nn.Sigmoid()(Sideoutput(2, kernel=7, factor=4)(enc))
                #pruned_side.append(enc)
            elif i ==2:
                enc = torch.cat([feat[:, 0:2, :, :], feat[:, 3:5, :, :], feat[:, 6:7, :, :], feat[:, 8:9, :, :],feat[:, 11:12, :,:]], dim=1)

                enc = torch.nn.Sigmoid()(Sideoutput(2, kernel=7, factor=8)(enc))
                #pruned_side.append(enc)
            elif i ==3:
                enc = torch.cat([feat[:, 4:5, :,:],
                                feat[:, 7:8, :, :],
                                 feat[:, 13:14, :, :],
                                 feat[:, 25:26, :, :],
                                 feat[:, 27:28, :, :]],
                                dim=1)
                enc = torch.nn.Sigmoid()(Sideoutput(2, kernel=7, factor=16)(enc))
                pruned_side.append(enc)
            """
                        elif i == 4:
                enc = torch.cat([feat[:, 0:1, :, :],
                                feat[:, 4:5, :, :],
                                 feat[:, 7:8, :, :],
                                 feat[:, 10:12, :, :],
                                 feat[:, 14:15, :, :],
                                 feat[:, 16:17, :, :],
                                 feat[:, 18:19, :, :],
                                 feat[:, 24:26, :, :],
                                 feat[:, 27:33, :, :],
                                 feat[:, 37:43, :, :],
                                 feat[:, 44:45, :, :],
                                 feat[:, 46:49, :, :],
                                 feat[:, 51:53, :, :],
                                 feat[:, 54:56, :, :],
                                 feat[:, 59:60, :, :],
                                 feat[:, 61:64, :, :]],dim=1)
                enc = torch.nn.Sigmoid()(Sideoutput(2, kernel=7, factor=32)(enc))
                pruned_side.append(enc)
            """

               # print(enc.size())

            side.append(b1)

        for i in range(0, len(self.filter_config)):

            unpool, feat = self.decoders[i](feat, indice[len(self.filter_config)-1-i])


        #xx = torch.cat([side[0], side[1], side[2], side[3], side[4], pruned_side[0], pruned_side[1],pruned_side[2],feat],dim=1)

        xx = torch.cat([ pruned_side[0],feat], dim = 1)
        #print(xx.size())
        xx = self.classifier1(xx)

        final = (xx)

        #print(final.size())
        return F.softmax(final, dim=2)


class Sideoutput(nn.Module):
    def __init__(self, filter, kernel, factor):

        super(Sideoutput, self).__init__()


        self.features = side_output(filter, kernel, factor)


            #if n_blocks == 3:
                #layers += [nn.Dropout(drop_rate)]


    def forward(self, x):
        output = self.features(x)


        return output

class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=1, kernelsize =7):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()


        if n_blocks  ==2:
            self.features= nn.Sequential(Upconv(n_in_feat,n_out_feat, kernel_size= kernelsize),
                                         Upconv(n_out_feat, n_out_feat, kernel_size= kernelsize)
            )
        elif n_blocks == 3:
            self.features = nn.Sequential(
                Upconv(n_in_feat, n_out_feat, kernel_size= kernelsize),
                Upconv(n_out_feat, n_out_feat, kernel_size= kernelsize),
                Upconv(n_out_feat, n_out_feat, kernel_size= kernelsize)
            )
        elif n_blocks==1:

            self.features = nn.Sequential(Upconv(n_in_feat, n_out_feat, kernel_size= kernelsize))



    def forward(self, x):
        output = self.features(x)

        #print(F.max_pool2d(output, 2, 2, return_indices=True))
        output, ind = F.max_pool2d(output, 2,2, return_indices = True)
        return output, ind#F.max_pool2d(output, 2, 2, return_indices=True), output.shape


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2,  kernelsize=7):
        super(_Decoder, self).__init__()
        if n_blocks  ==2:
            self.features= nn.Sequential(Upconv(n_in_feat,n_out_feat, kernel_size= kernelsize),
                                         Upconv(n_out_feat, n_out_feat, kernel_size= kernelsize))
        elif n_blocks == 3:
            self.features = nn.Sequential(
                Upconv(n_in_feat, n_out_feat, kernel_size= kernelsize),
                Upconv(n_out_feat, n_out_feat, kernel_size= kernelsize),
                Upconv(n_out_feat, n_out_feat, kernel_size= kernelsize)
            )
        elif n_blocks==1:
            self.features = nn.Sequential(Upconv(n_in_feat, n_out_feat, kernel_size= kernelsize))


    def forward(self, x, ind):#, indices, size):
        unpooled = F.max_unpool2d(x, indices=ind, kernel_size=2)
        return unpooled,self.features(unpooled)

##model = SegNetHEDKernel7PrunedSpalling(num_classes=3, filter_config=(4, 8, 16, 32, 64), kernel_size=7, padding = 3).cuda()

#summary(model, (3,256,256))