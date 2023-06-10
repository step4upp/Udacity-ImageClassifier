# Imports
import argparse
import torch
from os import path
from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, transforms, models

# Suggestion from Image Classifier - Part 2 -Command Line App : Make functions
def argparsing():
    # Starting Network setting explaination
    parser = argparse.ArgumentParser(description="This is N_Network settings")
    
    parser.add_argument('data_dir',action='store')
    
    parser.add_argument('--arch',
                        help='Please select torchvision model',
                        type=str)
    
    parser.add_argument('--save_dir',
                        type=str,
                        help='[Mendatory] Please indicate save dicrectory(str)')
    
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Please set learning rate for the gradient(int)')
    
    parser.add_argument('--hidden_units',
                        type=int,
                        help='Please set hidden layers for feedforward(int)')
    
    parser.add_argument('--epochs',
                        type=int,
                        help='Please enter the number of epochs(int)')
    
    parser.add_argument('--gpu',
                        type=str,
                        help='Please enter "Y" for using GPU and Cuda')
    
    return parser.parse_args()

# Transforming the pictures in directory file for use
def transformer(train_dir, test_dir, valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.486,0.456,0.406],
                                                                [0.229,0.224,0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.486,0.456,0.406],
                                                               [0.229,0.224,0.225])])
    
    train_data = datasets.ImageFolder(train_dir, train_transforms)
    test_data = datasets.ImageFolder(test_dir, test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, test_transforms)
    
    return train_data, test_data, valid_data

# Creating data for iteration
def loader(train_data, test_data, valid_data):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=64)
    
    return train_loader, test_loader, valid_loader

# Setting the calculation method
def gpu_select(gpu_argument):
    if type(gpu_argument) == type(None):
        print('Not using Cuda and GPU: Initiate CPU mode')
        return torch.device('cpu')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return device

def model_architecture(architecture):
    if type(architecture) == type(None):
        model = models.vgg16(pretrained=True)
        model.arch = "vgg16"
        print("Set to the default architecture: vgg16")
        
    else :
        model = eval("models.{}(pretrained=True)".format(architecture))
        model.arch = architecture
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def Classifier(model, hidden_units):
    if type(hidden_units) == type(None):
        hidden_units = 4096
        print("Setting to the default hidden units: 4096")
    
    # The in_feature is the number of inputs of the linear layer
    # model.classifier[0] means the first element of the classifier: nn.Linear(i,h)
    # The input number can be extracted by indicating specific component: in_features
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    input_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    
    return classifier

def training_setup(learning_rate, classifier):
    if type(learning_rate) == type(None):
        learning_rate = 0.001
        print("No learning rate input: default rate = 0.001")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    
    return criterion, optimizer

def training_network(model, train_loader, valid_loader, device,
                     criterion, optimizer, epochs, print_every, steps):
    
    if type(epochs) == type(None):
        epochs = 5
        print("Underdoing with default epoch: 5")
        
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        for inputs, labels in train_loader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0
    
                for inputs,labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    output = model.forward(inputs)
                    valid_loss += criterion(output, labels).item()

                    ps = torch.exp(output)
                    top_p , top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Eploch {epoch+1}/{epochs}"
                      f"Train loss: {running_loss/print_every:.3f}"
                      f"Test loss: {valid_loss/len(valid_loader):.3f}"
                      f"Test accuracy: {accuracy/len(valid_loader):.3f}")
                
                running_loss = 0
                model.train()
    
    print("Training finished")            
    return model

def model_validation(model, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
            
    print("Accuracy on test images: {:.3f}".format(100 * correct / total))

def checkpoint(model, save_dir, train_data):
    if type(save_dir) == type(None):
        print("No saving directory found")
    else:
        if path.isdir(save_dir):
            model.class_to_idx = train_data.class_to_idx
            
            checkpoint = {'architecture': model.arch,
                          'classifier': model.classifier,
                          'class_to_idx': model.class_to_idx,
                          'state_dict':model.state_dict()}
            
            torch.save(checkpoint, save_dir + '/checkpoint.pth')
            
        else:
            print("No file found in the directory: Unable to save")
            
def main():
    argument = argparsing()
    
    data_dir = argument.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data, test_data, valid_data = transformer(train_dir, test_dir, valid_dir)
    
    train_loader, test_loader, valid_loader = loader(train_data, test_data, valid_data)
    
    model = model_architecture(argument.arch)
    
    model.classifier = Classifier(model, argument.hidden_units)
    
    device = gpu_select(argument.gpu)
    
    model.to(device);
    
    criterion, optimizer = training_setup(argument.learning_rate, model.classifier)
    
    print_every = 25
    steps = 0
    
    model_trained = training_network(model, train_loader, valid_loader, device, criterion,
                                     optimizer, argument.epochs, print_every, steps)
    
    model_validation(model_trained, test_loader, device)
    
    checkpoint(model_trained, argument.save_dir, train_data)
    
if __name__ == '__main__':
    main()