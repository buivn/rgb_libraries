import math
import numbers
import random

from PIL import Image, ImageOps
import numpy as np
from torchvision import datasets, models,transforms
import math
# import  JointTransform  as joint_transforms


def rotate(x,y,xo,yo,theta): #rotate x,y around xo,yo by theta (rad)
    xr=math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo)   + xo
    yr=math.sin(theta)*(x-xo)+math.cos(theta)*(y-yo)  + yo
    return [xr,yr]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, cropflag = False):

        if cropflag == True:
            return img, mask
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        #print(w)
        if (w-tw)>0 and (h-th)>0:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            #print("x="+str(x1))
            #print("y="+str(y1))
            img = img.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))

            rand = random.randint(0,5)


        
            if rand == 0 or rand ==4 or rand ==5 or rand== 6:
                img = img.rotate(90)
                mask = mask.rotate(90)
            elif rand == 1:
                img, mask = joint_transforms.RandomVerticallyFlip()(img, mask)
            elif rand == 2:
                img, mask = joint_transforms.RandomGammaCorrection()(img, mask)
            if rand == 3:
                img, mask = joint_transforms.RandomHorizontallyFlip()(img, mask)

            """
            if rand == 0:
                img, mask = joint_transforms.RandomHorizontallyFlip()(img,mask)
            elif rand == 1:
                img, mask = joint_transforms.RandomVerticallyFlip()(img, mask)
            #img, mask = joint_transforms.RandomRotate(random.randint(0,360))(img, mask)
            elif rand ==2:
                img, mask = joint_transforms.RandomGammaCorrection()(img,mask)
            elif rand >=3 and rand<5:
                img = img.rotate(90)
                mask = mask.rotate(90)#joint_transforms.RandomRotate(img, mask)
            """

        return img, mask#np.transpose((np.array(img)),(2,1,0)), np.transpose((np.array(mask)), (2,1,0))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class RandomVerticallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

class RandomRotate(object):
    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * 90- 90
        return transforms.functional.rotate(img, rotate_degree, Image.BILINEAR, expand =True), transforms.functional.rotate(mask,rotate_degree, Image.NEAREST, expand = True)



class RandomGammaCorrection(object):
    def __call__(self, img, mask):

        if random.random()<0.5:
            rand = random.random()
            return transforms.functional.adjust_gamma(img,rand,1), mask
        return img, mask

