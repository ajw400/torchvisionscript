import torch
import torchvision
from torchvision import models
import argparse
import utils
import model_utils
import json

#take arguments from the command line and store in args array
parser = argparse.ArgumentParser(description='Train a deep learning model with a dataset of your choice!')
parser.add_argument('data_dir', help='data directory path', nargs=1)
parser.add_argument('--save_dir', nargs='?', default=None,
                    help='specify a directory where you\'d like to save the checkpoint')
parser.add_argument('--arch', nargs='?', choices=['vgg13', 'densenet121'], default='densenet121',
                    help='specify the architecture you would like to use')
parser.add_argument('--learning_rate', nargs='?', type=float, default=0.003, help='specify the learning rate to use')
parser.add_argument('--epochs', nargs='?', default=10, type=int, help='over how many epochs would you like to train')
parser.add_argument('--hidden_units', nargs='?', default=512, type=int, help='specify the size of the hidden layer')
parser.add_argument('--gpu', default=False, action='store_true', help='add this flag to train on GPU!')
args = parser.parse_args()

print("Creating model....")
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
dataloaders = utils.create_dataloaders(args.data_dir[0])

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
output_length = len(cat_to_name)

model, criterion, optimizer = model_utils.build_model(args, output_length, device)
print("Model built successfully!")
print("Training model...")
model_utils.train_model(args, model, criterion, optimizer, dataloaders, device)
print("Model training finished!")
print("Testing model...")
model_utils.test_model(model, criterion, dataloaders, device)
# model_utils.save_model