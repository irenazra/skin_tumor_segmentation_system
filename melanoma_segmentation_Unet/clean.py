# Simple script use to clean the data that is downloaded from the ISIC site

# Import depedencies
import os

for file in os.listdir('./data/Training'):
    if file.endswith('.png') or file.endswith('.csv'): # png and csv files in the training data are not needed
        os.remove('./data/Training/'+file)

for file in os.listdir('./data/Masks'):
    if file.endswith('.csv'): # csv files in the validation data are not needed
        os.remove('./data/Masks/'+file)
