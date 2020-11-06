class dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        #get all the images in a list (jpg extension)
        #get all the targets in a list (png extension)
        self.length = #length of the image or target list
        self.transformation = #this will be from the data_prep file

    def __len__ (self):
        return self.length

    def __getitem__(self,idx):
        #get the idxth image and the target from the lists
        #Perform the desired transformation

        #return a tuple of (image,target)