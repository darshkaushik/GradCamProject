'''
This is a refernce of how to use the custom dataset
We have to override just two methods named "len" and getitem
Docstrings of the functions has been given in the code itself.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    '''
    csv_file is the csv_file containing the information
    root_dir is the parent directory/folder containg the images
    transforms specifies the transformation to be applied on the images
    '''

    def __init__(self,csv_file,root_dir,transforms=None):
        self.csv_file=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transforms=transforms

    def __len__(self):
        """
        :return: returns the length of the dataset
        """

        return len(self.csv_file)

    def __getitem__(self, item):

        """

        :param item:it is the the index of the data item to be fetched.
        :return: it returns the specified index dataitem and its corresponding label

        """

        image_path = os.path.join(self.root_dir,self.csv_file.iloc[item,0])
        image = Image.open(image_path)
        row=self.csv_file.iloc[item,-3:-1]
        label=torch.tensor(row)

        if self.transforms:
            image=self.transforms(image)

        return (image,label)





