from os import listdir
import torch 
from PIL import Image
class dataset(torch.utils.data.Dataset):
    def __init__(self,transforms):
        super().__init__()

        #get all the images in a list (jpg extension)
        #get all the targets in a list (png extension)
        self.images =listdir('data/Training')  # TODO : GET CORRECT PAHTS
        self.targets = listdir('data/Masks') # TODO : GET CORRECT PAHTS
        self.length = len (self.images)
        self.transforms = transforms
    def __len__ (self):
        return self.length

    def __getitem__(self,i):
        #get the idxth image and the target from the lists
        if type(self.images[i]) == type([]):
            image_name = 'data/Training/'+self.images[i][0]
            target_name = 'data/Masks/'+self.targets[i][0]
        else:
            image_name = 'data/Training/'+self.images[i]
            target_name = 'data/Masks/'+self.targets[i]

        image=Image.open(image_name).convert('RGB') # TODO : GET CORRECT PAHTS
        target = Image.open(target_name).convert('RGB') # TODO : GET CORRECT PAHTS

        #Perform the desired transformation
        image, target = self.transforms(image,target)

        #return a tuple of (image,target)
        return (image,target)