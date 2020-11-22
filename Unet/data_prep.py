# File the creates the necessary data transforms

# Import depedencies
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision

# Puts all the transforms together and applies them
class PerformTransforms(object): # callable class
    def __init__(self, transforms):
        self.transforms = transforms
  
    def __call__(self, image, target):
        for trans in self.transforms: # apply transform in list of transforms
            image, target = trans(image, target)
        return image, target

# Turns the image into a tensor
class ImageTargetToTensor(object): # callable class
    def __call__(self, image, target):
        transform = torchvision.transforms.ToTensor()
        image = transform(image)
        target = transform(target)
        return image, target


# Resizes the image and the target
class ImageTargetResize(object): # callable class
    def __init__(self, size): 
        self.size = size # establish the image size for tranform

    def __call__(self, image, target): # apply resize given image and target
        size_transform = torchvision.transforms.Resize(self.size)
        image = size_transform(image)
        target = size_transform(target)
        return image, target

