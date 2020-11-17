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
import torchvision
import PIL
from PIL import Image  

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
        img = torchvision.utils.make_grid(torch.cat((x,y,y_hat)), nrow=x.shape[0], padding=10)
        path  = './Preds/'
        name = path+str(batch_idx)+".jpg"
        torchvision.utils.save_image(img, name)

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

        num_data = len(data_set)
        indices = torch.randperm(num_data).tolist()
        training_ratio = 0.7
        training_number = int(training_ratio * num_data)

        self.training_data = torch.utils.data.Subset(
                data_set, indices[0:training_number])
        self.validation_data = torch.utils.data.Subset(
                data_set, indices[training_number:])
        
   
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.TEST_params.batch_size,shuffle=True) 

 
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data, batch_size=self.TEST_params.batch_size,shuffle=True) 


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

    
