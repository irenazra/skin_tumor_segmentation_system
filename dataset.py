import glob
import torch 
from PIL import Image
class dataset(torch.utils.data.Dataset):
    def __init__(self,transforms):
        super().__init__()

        #get all the images in a list (jpg extension)
        #get all the targets in a list (png extension)
        self.images = glob.glob('data/*.jpg')  # TODO : GET CORRECT PAHTS
        self.targets = glob.glob('data/*.png') # TODO : GET CORRECT PAHTS
        self.length = len (self.images)
        self.transforms = transforms
    def __len__ (self):
        return self.length

    def __getitem__(self,i):
        #get the idxth image and the target from the lists
        image_name = self.images[i]
        target_name = self.targets[i]

        image=Image.open('data/ISIC_0000028 copy.jpg').convert('RGB') # TODO : GET CORRECT PAHTS

        #target = Image.open(target_name).convert('RGB')
        target = Image.open('data/ISIC_0000028_segmentation copy.png').convert('RGB') # TODO : GET CORRECT PAHTS

        #Perform the desired transformation
        image, target = self.transforms(image,target)

        #return a tuple of (image,target)
        return (image,target)