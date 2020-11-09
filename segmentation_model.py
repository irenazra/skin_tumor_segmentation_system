import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import Unet 
import dataset
import data_prep

class segmentation_model (LightningModule):
    def __init__(self,parameters):
        super().__init__()
        #initialize the unet model here
        self.model = Unet(3,3,256)
        self.parameters = parameters


    def forward(self,x):
        #pass x through the unet model defined above an return the result
        return self.model(x)

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
        test_loss = F.cross_entropy(y_hat,y)
        return test_loss

    #Lightning calls this at the beginning, only once
    def prepare_data(self):
        trans = data_prep.PerformTransforms([
            data_prep.ImageTargetResize(
                (self.parameters.image_size, self.parameters.image_size)),
            data_prep.ImageTargetToTensor()
        ])
    
        dataset = dataset(trans)
        #self.test_data = we can make this come from a different folder

        num_data = len(dataset)
        training_ratio = 0.7
        training_number = training_ratio * num_data

        self.training_data = dataset[0:training_number]
        self.validation_data = dataset[training_number:(num_data - 1)]
        
   
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.parameters.batch_size,shuffle=True) 

 
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data, batch_size=self.parameters.batch_size,shuffle=True) 

    
    #def test_dataloader(self):
        #return torch.utils.data.DataLoader(self.test_data, batch_size=self.parameters.batch_size,shuffle=True) 

    # We might need to add methods to save images aand log val loss etc
    # TODO: IMPLEMENT THESE METHODS



    
