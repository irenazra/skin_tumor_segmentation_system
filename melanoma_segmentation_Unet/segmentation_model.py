# Main file that runs the entire model when called

# Import depedencies
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

class segmentation_model (LightningModule): # class  that wraps the entire model
    def __init__(self, parameters):
        super().__init__()
        # initialize the unet model here
        self.model = Unet.Unet(3,3,256)
        self.TEST_params = parameters
        self.image_size = 256

    # Pass x through the Unet defined above an return the result
    def forward(self,x):
        return self.model(x)

    # Set the optimizer
    def configure_optimizers(self):
        # Optimizer = Adam
        # Learning rate = 1e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        return optimizer

    # Determine how to conduct a training step
    def training_step(self,batch,batch_idx):
        x,y = batch  # get an image and target
        y_hat = self(x) # create prediction
        loss = F.binary_cross_entropy_with_logits(y_hat,y) # evaluate using cross-entropy loss
        return loss # return training loss

    # Determine how to conduct a validation step (similar to training step)
    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        val_loss = F.binary_cross_entropy_with_logits(y_hat,y)

        # Save validation batch results for access later
        # Make 3-row image with the actual image, target, and prediction
        img = torchvision.utils.make_grid(torch.cat((x,y,y_hat)), nrow=x.shape[0], padding=10)
        path  = './Preds/'
        name = path+str(batch_idx)+".jpg" # custom name based on batch index
        torchvision.utils.save_image(img, name) # save image

        return val_loss # return validation loss

    # Determine how to conduct a test  step
    def test_step(self,batch,batch_idx):
        x,y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat,y)
        return test_loss

    # Lightning calls this at the beginning only once
    # Perpares data by applying transforms and makes the dataset
    # Uses dataset and data_prep files
    def prepare_data(self):
        trans = data_prep.PerformTransforms([
            data_prep.ImageTargetResize(
                (self.image_size, self.image_size)),
            data_prep.ImageTargetToTensor()
        ]) # set tranforms
    
        data_set = dataset.dataset(trans) # create transformed dataset

        # Split the data using a 70-30 training-test split
        num_data = len(data_set)
        indices = torch.randperm(num_data).tolist()  # create list of indices from a shuffle of the dataset
        training_ratio = 0.7
        training_number = int(training_ratio * num_data) # number of images we'll use to train

        self.training_data = torch.utils.data.Subset(
                data_set, indices[0:training_number]) # use 70% of images to train
        self.validation_data = torch.utils.data.Subset(
                data_set, indices[training_number:])  # use 30% of images to validate
        
   
    # Torch data_loader for training data
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.TEST_params.batch_size,shuffle=True) 

    # Torch data_loader for validation data
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data, batch_size=self.TEST_params.batch_size,shuffle=True) 


# Main method
if __name__ == "__main__":
    # Optional argument passing for batch size and image size
    parser = argparse.ArgumentParser(description='UI Segmentation Training')
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument("--image_size", type=int, default=256,
                        help="size of training images, default is 256")
    args = parser.parse_args()

    # Define and tain model
    model = segmentation_model(args)
    trainer = Trainer(max_epochs=5)
    trainer.fit(model)

    
