import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import numpy

class Block():
    def __init__(self, image):
        super(Block, self).__init__()
        self.image = image
        self.img = image.load()
    def accuracy(self, gt):

        groundtruth = gt.load()
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(0, self.image.size[0]):
            for j in range(0, self.image.size[1]):
                if self.img[i,j] == (0,0,0):
                    if groundtruth[i,j] == (0,0,0):
                        TN = TN + 1
                    else:
                        FN = FN + 1
                else:
                    if groundtruth[i,j] == (255,255,255):
                        TP = TP + 1
                    else:
                        FP = FP + 1

        return [TP, FP, TN, FN]

class Threshold():
    def __init__(self, predicted):
        super(Threshold, self).__init__()

        self.pred = predicted *100

    def threshold(self, th, skip=3):
        thresh = numpy.zeros([self.pred.shape[0], self.pred.shape[1], 3])
        for i in range(0+skip, thresh.shape[0]-skip):
            for j in range(0, thresh.shape[1]):
                if self.pred[i, j]  >= th:
                    thresh[i, j, :] = 255
        return thresh
    def threshold_multiclass(self, th, skip=3):
        thresh = numpy.zeros([self.pred.shape[0], self.pred.shape[1], 3])
        for i in range(0+skip, thresh.shape[0]-skip):
            for j in range(0, thresh.shape[1]):
                if self.pred[i, j, 0]  >= th and self.pred[i, j, 1]<=th and self.pred[i, j, 2] <=th:
                    thresh[i, j, :] = [255,0,0]
                if self.pred[i, j, 0]  >= th and self.pred[i, j, 1]>=th and self.pred[i, j, 2] >=th:
                    thresh[i, j, :] = [255,255,255]

        return thresh

class Superimpose():
    def __init__(self, image, groundtruth, predicted):
        super(Superimpose, self).__init__()

        self.img = image
        self.gt = groundtruth
        self.pred = predicted

    def superimpose_result(self):

        img1 = self.img
        TP, P_GT = 0, 0
        FP, N_GT = 0, 0
        FN = 0
        TN = 0
        list = []
        for i in range(0, self.img.shape[0]):
            for j in range(0, self.img.shape[1]):
                if self.gt[i][j][0] == 255 and self.gt[i][j][1] == 255 and self.gt[i][j][2] == 255:
                    P_GT = P_GT + 1
                    if self.pred[i][j][0] == 255 and self.pred[i][j][1] == 255 and self.pred[i][j][2] == 255:
                        img1[i, j, :] = [255, 0, 0]  # TP = RED
                        TP = TP + 1

                    else:
                        img1[i, j, :] = [0, 255, 0]  # FN = GREEN
                        FN = FN + 1
                else:
                    N_GT = N_GT + 1
                    if self.pred[i][j][0] == 255 and self.pred[i][j][1] == 255 and self.pred[i][j][2] == 255:
                        img1[i, j, :] = [0, 0, 255]  # FP = BLUE
                        FP = FP + 1

                    else:
                        img1[i, j, :]  # TN = AS IT IS
                        TN = TN + 1

        # list.append(TP)
        # list.append(TP)
        #list.append(img1)
        #list.append(TP)
        #list.append(FP)
        #print(img1.shape)
        return img1, [TP, FP, TN, FN, P_GT, N_GT]



    def superimpose_result_with_threshold(self):

        img1 = self.img
        TP, P_GT = 0, 0
        FP, N_GT = 0, 0
        FN = 0
        TN = 0
        list = []
        for i in range(0, self.img.shape[0]):
            for j in range(0, self.img.shape[1]):
                if self.gt[i][j]== 255:
                    P_GT = P_GT + 1
                    if self.pred[i][j][0] == 255 and self.pred[i][j][1] == 255 and self.pred[i][j][2] == 255:
                        img1[i, j, :] = [255, 0, 0]  # TP = RED
                        TP = TP + 1

                    else:
                        img1[i, j, :] = [0, 255, 0]  # FN = GREEN
                        FN = FN + 1
                else:
                    N_GT = N_GT + 1
                    if self.pred[i][j][0] == 255 and self.pred[i][j][1] == 255 and self.pred[i][j][2] == 255:
                        img1[i, j, :] = [0, 0, 255]  # FP = BLUE
                        FP = FP + 1

                    else:
                        img1[i, j, :]  # TN = AS IT IS
                        TN = TN + 1

        # list.append(TP)
        # list.append(TP)
        #list.append(img1)
        #list.append(TP)
        #list.append(FP)
        #print(img1.shape)
        return img1, [TP, FP, TN, FN, P_GT, N_GT]#, (TP/P_GT)*100,(FP/ N_GT) *100, (TN/N_GT)*100, (FN/P_GT) *100]

class PILImageUtility():
    def __init__(self, original, label, predicted):
        super(PILImageUtility, self).__init__()

        self.pred = predicted#.load()
        self.label = label
        self.orig  = original

    def superimpose_result(self, threshold):

        predicted_pixel = self.pred.load()
        ground_truth = self.label.load()
        Label = self.orig
        #Label = Image.new(mode='RGB', size=self.orig.size)
        labelpixel = Label.load()
        TP, P_GT = 0, 0
        FP, N_GT = 0, 0
        FN = 0
        TN = 0
        list = []

        for i in range(self.pred.size[0]):  # for every pixel:
            for j in range(self.pred.size[1]):
                if  ground_truth[i,j] ==255:
                    P_GT = P_GT+1
                    if predicted_pixel[i,j] >= threshold:
                        TP = TP+1
                        labelpixel[i,j] = (255,0,0)
                    else:
                        FN = FN+1
                        labelpixel[i, j] = (0, 255, 0)
                else:
                    N_GT = N_GT+1
                    if predicted_pixel[i,j] >= threshold:
                        FP = FP+1
                        labelpixel[i, j] = (0, 0, 255)
                    else:
                        TN = TN+1
            # list.append(TP)
            # list.append(TP)
            # list.append(img1)
            # list.append(TP)
            # list.append(FP)
            # print(img1.shape)
        return Label, [TP, FP, TN, FN, P_GT, N_GT, (TP/P_GT)*100,(FP/ N_GT) *100, (TN/N_GT)*100, (FN/P_GT) *100 ]

        #self.converted_label =  self.convert_threshold_pillow(threshold=0.8)
        #self.image_list = [original, label, self.converted_label]
    """
        def convert_threshold_pillow(self, threshold):
        pixels = self.img.load()  # create the pixel map
        Label = Image.new(mode='RGB', size=self.img.size)
        labelpixel = Label.load()


        for i in range(self.img.size[0]):  # for every pixel:
            for j in range(self.img.size[1]):
                if pixels[i, j] >= (threshold, threshold, threshold):
                    labelpixel[i, j] = (255, 255, 255)

        #self.image_list.append(Label)
        #self.image_list.append(Label)
        return Label
    """
    def convert_2dto3d_pillow(self, size):

        Label = Image.new(mode='RGB', size=size)
        labelpixel = Label.load()
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                if self.pred[0, i, j] > self.pred[1,i,j]:
                    labelpixel[i,j] = (255,255,255)
                else:
                    labelpixel[i, j] = (0, 0, 0)
        return  Label


class PILImageSave():
    def __init__(self, original, labelnew, predicted):
        super(PILImageSave, self).__init__()
        self.image_list = [original,labelnew, predicted]
        self.label = labelnew
        self.pred = predicted

    def threshold(self, thresh):
        threshold = Image.new(mode='RGB', size = self.label.size)
        threshold_pixel = threshold.load()
        predicted_pixel = self.pred.load()
        for i in range(0, self.label.size[0]):
            for j in range(0, self.label.size[1]):
                if predicted_pixel[i, j] > thresh:
                    threshold_pixel[i,j] = (255,255,255)
                else:
                    threshold_pixel[i, j] = (0, 0, 0)
        self.image_list.append(threshold)


    def save_images(self, w=512):

        total_width = len(self.image_list) * w
        total_height = w + 20
        new_im = Image.new('RGB', (total_width, total_height))
        x_offset = 0
        for im in self.image_list:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        #draw = ImageDraw.Draw(new_im)

        #draw.text((512, 512 + 10), "original--->label--->predictedtOriginal--->Thresholded" + str(loss),
        #          (255, 255, 255))
        return new_im

class ConvertBinary():
    def __init__(self, image):
        super(ConvertBinary, self).__init__()
        self.image = image

    def convert_binary(self):
        label = numpy.zeros([self.image.shape[0], self.image.shape[1]], dtype='uint8')

        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                R = self.image[i][j][0]
                G = self.image[i][j][1]
                B = self.image[i][j][2]

                if R == 255 and G == 255 and B == 255:
                    label[i][j] = 1
        return label

    def convert_binary_pillow(self):
        pixels = self.image.load()  # create the pixel map
        Label = Image.new(mode='1', size=self.image.size)
        labelpixel = Label.load()

        for i in range(self.image.size[0]):  # for every pixel:
            for j in range(self.image.size[1]):
                if pixels[i, j] == (255, 255, 255):
                    labelpixel[i, j] = 255
                else:
                    labelpixel[i,j] = 0
        return Label

    def convert_binary_pillow_multilabel(self):
        pixels = self.image.load()  # create the pixel map
        Label = Image.new(mode='1', size=self.image.size)
        labelpixel = Label.load()

        for i in range(self.image.size[0]):  # for every pixel:
            for j in range(self.image.size[1]):
                if pixels[i, j] == (255, 255, 255):
                    labelpixel[i, j] = 255
                else:
                    labelpixel[i,j] = 0
        return Label

    def convert_binary_pillow_3d_to_2d(self):
        pixels = self.image.load()  # create the pixel map
        Label = Image.new(mode='1', size=[2,self.image.size[0], self.image_size[1]])
        labelpixel = Label.load()

        for i in range(self.image.size[0]):  # for every pixel:
            for j in range(self.image.size[1]):
                if pixels[i, j] == (255, 255, 255):
                    labelpixel[i, j] = 1
        return Label