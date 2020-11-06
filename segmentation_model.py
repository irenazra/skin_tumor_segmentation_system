import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

class segmentation_model (LightningModule):
    def __init__(self):
        super().__init__()
        #initialize the unet model here

    def forward(self,x):
        #pass x through the unet model defined above an return the result
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer

    def training_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat,y)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat,y)
        return val_loss

    def test_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat,y)
        return loss

    #Lightning calls this at the beginning, only once
    # TODO: IMPLEMENT THIS METHOD
    #def prepare_data(self):

        #self.dataset = dataset.py 

        #self.test_data = you can make this come from a different folder etc

        #To assign the following, we can divide the image, target lists into two parts
        #self.validation_data
        #self.training_data

    #these methods will probably just return the lists that we create in prepare_data
    # TODO: IMPLEMENT THIS METHOD
    #def train_dataloader(self):

    # TODO: IMPLEMENT THIS METHOD
    #def val_dataloader(self):

    # TODO: IMPLEMENT THIS METHOD
    #def test_dataloader(self):


    # We might need to add methods to save images aand log val loss etc
    # TODO: IMPLEMENT THESE METHODS



    
