
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision

#turns the image into a tensor
class ImageTargetToTensor(object):
    def __call__(self, image, target):
        transform = torchvision.transforms.ToTensor()
        image = transform(image)
        target = transform(target)
        return (image, target)


#resizes the image and the target
class ImageTargetResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        size_transform = torchvision.transforms.Resize(self.size)
        image = size_transform(image)
        target = size_transform(target)
        return (image, target)


if __name__ == "__main__":
    
    image = Image.open("data/ISIC_0000028 copy.jpg")
    target = Image.open("data/ISIC_0000028_segmentation copy.png")
    resizer = ImageTargetResize(200)
    print(resizer.__call__(image,target))
