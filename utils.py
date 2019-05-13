import torch
import numpy as np
from torchvision import datasets, transforms
means = [0.485, 0.456, 0.406]
stdev = [0.229, 0.224, 0.225]

#takes a root path of data with subdirectory structure consisting of /train /valid and /test and returns dataloaders
def create_dataloaders(filepath):
    #create directories
    train_dir = filepath + '/train'
    valid_dir = filepath + '/valid'
    test_dir = filepath + '/test'
    
    #specify transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                         transforms.Normalize(means, stdev)])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, stdev)])
    #load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    
    #create and return data loaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    
    return {'train': train_dataloader, 'validate': valid_dataloader, 'test': test_dataloader}
