import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import pytorch_lightning
import Unet 
import dataset
import data_prep
import argparse 

class segmentation_model (LightningModule):
    def __init__(self, parameters):
        super().__init__()
        #initialize the unet model here
        self.model = Unet.Unet(3,3,256)
        self.TEST_params = parameters
        self.image_size = 256


    def forward(self,x):
        #pass x through the unet model defined above an return the result
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        return optimizer

    def training_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat,y)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        val_loss = F.binary_cross_entropy_with_logits(y_hat,y)
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
                (self.image_size, self.image_size)),
            data_prep.ImageTargetToTensor()
        ])
    
        data_set = dataset.dataset(trans)
        #self.test_data = we can make this come from a different folder

        num_data = len(data_set)
        training_ratio = 0.7
        training_number = int(training_ratio * num_data)
        #self.training_data = data_set[0:training_number]
        self.training_data = torch.utils.data.Subset(
                data_set, [0])
        #self.validation_data = data_set[training_number:(num_data - 1)]
        #self.validation_data = data_set[1:] # Crappy version 
        self.validation_data = torch.utils.data.Subset(
                data_set, [1])
        
   
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.TEST_params.batch_size,shuffle=True) 

 
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data, batch_size=self.TEST_params.batch_size,shuffle=True) 

    
    #def test_dataloader(self):
        #return torch.utils.data.DataLoader(self.test_data, batch_size=self.parameters.batch_size,shuffle=True) 

    # We might need to add methods to save images aand log val loss etc
    # TODO: IMPLEMENT THESE METHODS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UI Segmentation Training')
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument("--image_size", type=int, default=256,
                        help="size of training images, default is 256")
    args = parser.parse_args()

    model = segmentation_model(args)
    trainer = Trainer(max_epochs=5)
    trainer.fit(model)

    
