import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
    
    return {'train': train_dataloader, 
            'validate': valid_dataloader, 
            'test': test_dataloader, 
            'class_to_idx':   train_dataset.class_to_idx}

def process_image(image_path):
    image = Image.open(image_path)
    x, y = image.size
    if x > y:
        x, y = 256 * x / y, 256
    else:
        x, y = 256, 256 * y / x  
    image.thumbnail((x,y))
    left = (x-224)/2
    top = (y-224)/2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image) / 255
    np_image = (np.subtract(np_image, means)) / stdev
    np_image = np_image.transpose((2, 0, 1))
    image = torch.from_numpy(np_image)
    image = image.unsqueeze(0).type(torch.FloatTensor)
    return image

def show_image(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))
    image = stdev * image + means
    image = np.clip(image, 0, 1)
    print(plt)
    ax.imshow(image)
    return ax