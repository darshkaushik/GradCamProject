from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import pandas as pd
import numpy as np


class SkinMnistDataset(Dataset):
    '''
    csv_file is the csv_file containing the information
    root_dir is the parent directory/folder containing the images
    transforms specifies the transformation to be applied on the images
    '''

    def __init__(self, csv_file, root_dir, transforms=None):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        '''
        :return: returns the length of the dataset
        '''
        return len(self.csv_file)

    def __getitem__(self, item):
        """
        :param item:it is the the index of the data item to be fetched.
        :return: it returns the specified index dataitem and its corresponding label
        """
        image_path = os.path.join(
            self.root_dir, self.csv_file['image_path'][item])
        image = Image.open(image_path)
        row = self.csv_file.iloc[item, -3:]
        label = torch.tensor(row)

        if self.transforms:
            image = self.transforms(image)
        return (image, label)


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.6373545, 0.44605875, 0.46191868], [
                             0.27236816, 0.22500427, 0.24329403])
    ]),
    'test': transforms.Compose([
        transforms.Resize(226),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.6373545, 0.44605875, 0.46191868], [
                             0.27236816, 0.22500427, 0.24329403])
    ]),
}
