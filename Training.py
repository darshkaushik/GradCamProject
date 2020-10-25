import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from CustomDatasetPytorch import CustomDataset

batch_size=8
shuffle=True

#Do remember to specify transformations.Use transforms.compose for more than one transformations
dataset=CustomDataset(csv_file="csv_file_name",root_dir="root_dir_image",transforms=None)

#the second parameter is the size of train and test set.Do specify as per dataset
train_set,test_set=random_split(dataset,[400,100])

#These are the train test generators
train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=shuffle)
test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=shuffle)

'''
Start training or coding from here i.e calling pretrained model,using callbacks,
forward prop & backward prop etc.

'''


