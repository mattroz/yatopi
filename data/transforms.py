import cv2
import torch
import numpy as np
import random
import warnings
import numbers
import math

from PIL import Image

import torchvision.transforms as transforms


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR, imagenet=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.imagenet = imagenet

    def __call__(self, image):
        img = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        img = self.toTensor(img)
        if self.imagenet:
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])(img)
        return {'image': img}


class ResizeStandardize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return {'image': img}


class ConvertToRGB(object):

    def __init__(self):
        pass

    def __call__(self, image):
        #assert type(image) == Image.Image, 'img argument should be PIL.Image'
        np_image = np.array(image)
        cvt_img = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        return {'image': cvt_img}


class RandomIntensity(object):

    def __init__(self, p=0.5, delta=(-0.05, 0.05)):
        self.p = p
        self.delta = delta
        #self.toTensor = transforms.toTensor()

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = img.clone()
            intensity_mask = np.random.uniform(self.delta[0], self.delta[1], img.shape)
            img += torch.Tensor(intensity_mask)
        return img


class MedianBlur(object):

    def __init__(self, p=0.5, k=3):
        self.p = p
        self.k = k

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = img.numpy()
            img = torch.Tensor(cv2.medianBlur(img, self.k))
        return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, value=0):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            # aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area)))
            w = h
            # w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.FloatTensor(np.random.normal(size=(img_c, h, w)))
                    # v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.FloatTensor(value).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, value=self.value)
            img = img.clone()
            img[:, x:x+h, y:y+w] = v
        return img
