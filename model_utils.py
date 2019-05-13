#returns a new model
from torchvision import models
from torch import nn, optim
import torch
from collections import OrderedDict
from workspace_utils import active_session

def build_model(args, output_size, device):
    # TODO: Build and train your network
    if args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        features_output_size = 1024
    elif args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        features_output_size = 25088
    else:
        print("Error - you have entered an invalid model. Please use -h to see the options!")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(features_output_size, args.hidden_units)), 
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(0.2)),
                                                ('fc3', nn.Linear(args.hidden_units, output_size)),
                                                ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    model.to(device)
    return model, criterion, optimizer

def train_model(args, model, criterion, optimizer, dataloaders, device):
    validate_every = 2
    running_loss = 0
    steps = 0
    with active_session():
        for epoch in range(args.epochs):
            for images, labels in dataloaders['train']:
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                steps += 1
                if steps % validate_every == 0:
                    with torch.no_grad():
                        model.eval()
                        accuracy = 0
                        valid_loss = 0
                        for images, labels in dataloaders['validate']:
                            optimizer.zero_grad()
                            images, labels = images.to(device), labels.to(device)
                            log_ps = model(images)
                            loss = criterion(log_ps, labels)
                            valid_loss += loss.item()

                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(dataloaders['validate']):.3f}.. "
                          f"Validation accuracy: {accuracy/len(dataloaders['validate']):.3f}")      
                        running_loss = 0
                        model.train()
                        
def test_model(model, criterion,  dataloaders, device)
    with torch.no_grad():
        testing_loss = 0
        accuracy = 0
        for images, labels in test_dataloader:
            model.eval()
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            testing_loss += loss.item()
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Test loss: {testing_loss/len(dataloaders['test']):.3f}.. "
                          f"Test accuracy: {accuracy/len(dataloaders['train']):.3f}")
    
    
