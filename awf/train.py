import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import tqdm
import os
import pickle
import warnings
warnings.filterwarnings("ignore") 

# find the available gpu device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
print ("Available cuda device:", device)

# parameters
num_epochs = 20
lr = 0.001
num_classes = 900
batch_size = 256
test_split = 0.15
val_split = 0.15
data_path = 


# function to load the data
def load_data():
    def categorize(labels, dict_labels=None):
        possible_labels = list(set(labels))
        possible_labels.sort()

        if not dict_labels:
            dict_labels = {}
            n = 0
            for label in possible_labels:
                dict_labels[label] = n
                n = n + 1

        new_labels = []
        for label in labels:
            new_labels.append(dict_labels[label])


        return new_labels
    
    npzfile = np.load(datapath)
    data = npzfile["data"]
    labels = npzfile["labels"]
    npzfile.close()
    
    data = data.reshape(data.shape[0], 1, data.shape[1])
    
    labels = categorize(labels)
    
    return np.array(data), np.array(labels)


# Dataset class
class Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dr1 = nn.Dropout(p = 0.25)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=4)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=4)
        
        self.lstm = nn.LSTM(311, 128)
        self.fc1 = nn.Linear(4096, nb_classes)
        

    def forward(self, inp):
        
        x = inp
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x,_ = self.lstm(x)
        
        x = x.view(batch_size,-1)
        
        x = self.fc1(x)
        
        x = F.softmax(x)
        
        return x
    
    
    
    
    
# Train and Test functions
def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%1000 == 0:
            print ("Loss: {:0.6f}".format(loss.item()))
    
def test(model, device, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            t = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(t.view_as(pred)).sum().item()
    # print ('Accuracy Reports:\n Number of correct predictions: {}, length of dataset: {} accuracy: {:.2f}\n'. \
    #        format(correct, len(loader.dataset), correct / len(loader.dataset)))
    
    return correct / len(loader.dataset)


# Loading the data
data, target = load_data()

# Splitting data into train, validation, and test
num_instances = data.shape[0]
num_cells = data.shape[2]
num_traces = int(num_instances / num_classes)

num = num_instances
indices = np.arange(num_instances)
np.random.shuffle(indices)

split = int(num_instances * (1 - test_split))
ind_test = np.array(indices[split:])

num = indices.shape[0] - ind_test.shape[0]
split = int(num * (1 - val_split))
ind_val = np.array(indices[split:num])
ind_train = np.array(indices[:split])

x_train, y_train = data[ind_train], target[ind_train]
x_valid, y_valid = data[ind_valid], target[ind_valid]
x_test, y_test = data[ind_test], target[ind_test]


# Wrapping data using data loaders
train_dataset = Data(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

valid_dataset = Data(x_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)

test_dataset = Data(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)



# initializing the model and optimizer
model = Net().to(device)
optimizer = optim.RMSprop(model.parameters(), lr = lr)



print ('-------- start training ...')
for epoch in range(num_epochs):
    print (f'----------------- Epoch {epoch} ------------------')
    train(model, device, train_loader, optimizer)
    acc = test(model, device, valid_loader)
    print (f'Validation accuracy: {acc*100:.2f}')
    
    
print ("====================================================================")

test_acc = test(model, device, test_loader)
print (f'Test accuracy: {acc*100:.2f}')
