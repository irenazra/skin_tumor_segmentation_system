import glob
from PIL import Image
class dataset(torch.utils.data.Dataset):
    def __init__(self,transforms):
        super().__init__()

        #get all the images in a list (jpg extension)
        #get all the targets in a list (png extension)
        self.images = glob.glob(f'{self.data}/*.png')
        self.targets = glob.glob(f'{self.data}/*.jpg')
        self.length = len (images)
        self.transformation = transforms
    def __len__ (self):
        return self.length

    def __getitem__(self,i):
        #get the idxth image and the target from the lists
        image_name = self.images[i]
        target_name = self.targets[i]

        image=Image.open(image_name).convert('RGB')
        target = Image.open(target_name).conver('RGB')

        #Perform the desired transformation
        image, target = self.transforms(image,target)

        #return a tuple of (image,target)
        return (image,target)