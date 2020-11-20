# Simple script use to clean the data that is downloaded from the ISIC site

# Import depedencies
import os

for file in os.listdir('./Training'):
    if file.endswith('.png'): # png files in the training data are not needed, so we remove the m
        os.remove('./Training/'+file)


