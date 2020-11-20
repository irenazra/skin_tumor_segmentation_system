# File that helps create the dataset for Torch Lightning

# Import depedencies
from os import listdir
import torch 
from PIL import Image

class dataset(torch.utils.data.Dataset): # torch dataset class
    def __init__(self,transforms):
        super().__init__()
        self.images =listdir('data/Training') #get all the images in a list (jpg extension)
        self.targets = listdir('data/Masks') #get all the targets in a list (png extension)
        self.length = len (self.images)
        self.transforms = transforms

    # len() returns method the number of images in the set
    def __len__ (self): 
        return self.length

    # Determine how is data gathered when the dataset is indexed
    def __getitem__(self,i):
        # Get the i-th image and the target from the lists
        # Sometimes path to data is returned as a list, sometimes as a single item so typechecking is needed 

        # case where paths are in a list
        if type(self.images[i]) == type([]):
            image_name = 'data/Training/'+self.images[i][0]
            target_name = 'data/Masks/'+self.targets[i][0]
        # case when paths are in a single item
        else:
            image_name = 'data/Training/'+self.images[i]
            target_name = 'data/Masks/'+self.targets[i]

        # Open images in RGB
        image=Image.open(image_name).convert('RGB')
        target = Image.open(target_name).convert('RGB')

        # Perform the desired transformation
        image, target = self.transforms(image,target)

        # Return a tuple of (image,target)
        return (image,target)