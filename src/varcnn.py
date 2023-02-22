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
batch_size = 256
num_epochs = 2
dir_dilations = True
time_dilations = True
scale_metadata = True
num_classes = 100
lr = 0.001
data_path = '/work/abahramali_umass_edu/wf_datasets/varcnn'



# Dataset class
class Data(Dataset):
    def __init__(self, x_dir, x_time, x_meta, y):
        self.x_dir = x_dir
        self.x_time = x_time
        self.x_meta = x_meta
        self.y = y
        
    def __getitem__(self, index):
        return self.x_dir[index], self.x_time[index], self.x_meta[index], self.y[index]
    
    def __len__(self):
        return len(self.x_dir)
        


# ---------- Model ----------
# Single ResNet block
class BasicBlockDilated(nn.Module):
    def __init__(self, inplanes, planes, stage=0, block=0, kernel_size=3, numerical_names=False, 
                 stride=None, dilations=(1,1)):
        super(BasicBlockDilated, self).__init__()
        self.block = block
        if stride is None:
            if block!=0 or stage==0:
                stride=1
            else:
                stride=2
        

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=dilations[0],
                              bias=False, dilation=dilations[0])
        self.bn1 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=dilations[1],
                              bias=False, dilation=dilations[1])

        self.bn2 = nn.BatchNorm1d(planes)

        if block == 0:
            self.conv3 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False)

            self.bn3 = nn.BatchNorm1d(planes)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.block == 0:
            identity = self.conv3(x)
            identity = self.bn3(identity)

        out += identity
        out = self.relu(out)
        
        return out
    
# Resnet model
class ResNet18(nn.Module):
    def __init__(self, blocks=None, block=None, numerical_names = None):
        super(ResNet18, self).__init__()
        
        
        inplanes = 1
        in_channels = 64
        
        if blocks is None:
            blocks = [2, 2, 2, 2]
        if block is None:
            block = BasicBlockDilated
        if numerical_names is None:
            numerical_names = [True] * len(blocks)
        
        self.conv1 = nn.Conv1d(inplanes, 64, kernel_size=7, stride=2, bias=False, padding=3)
        
        self.bn1 = nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layers = []
        
        out_channels = 64
        
        self.dilated_layers = nn.Sequential(
        BasicBlockDilated(64, 64, 0, 0, dilations=(1,2), numerical_names = False),
        BasicBlockDilated(64, 64, 0, 1, dilations=(4,8), numerical_names = False),
        
        BasicBlockDilated(64, 128, 1, 0, dilations=(1,2), numerical_names = False),
        BasicBlockDilated(128, 128, 1, 1, dilations=(4,8), numerical_names = False),
        
        BasicBlockDilated(128, 256, 2, 0, dilations=(1,2), numerical_names = False),
        BasicBlockDilated(256, 256, 2, 1, dilations=(4,8), numerical_names = False),
        
        BasicBlockDilated(256, 512, 3, 0, dilations=(1,2), numerical_names = False),
        BasicBlockDilated(512, 512, 3, 1, dilations=(4,8), numerical_names = False)
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        
        out = self.dilated_layers(out)
    
        out = self.avgpool(out)
        
        return out
        
        
# Combined model for time, direction, and statistical features
class VarCNN(nn.Module):
    def __init__(self):
        super(VarCNN, self).__init__()
        self.resnet_time = ResNet18()
        self.resnet_dir = ResNet18()
        
        self.fc_met = nn.Linear(7, 32)
        self.bn_met = nn.BatchNorm1d(32)
        self.relu_met = nn.ReLU()
        
        self.fc = nn.Linear(1056, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
        self.last_fc = nn.Linear(1024, num_classes)
        
    def forward(self, dir_input, time_input, metadata_input):
        
        t = self.resnet_time(time_input)
        d = self.resnet_dir(dir_input)
        m = self.fc_met(metadata_input)
        m = self.bn_met(m)
        m = self.relu_met(m)
        
        t = t.view((-1, 512))
        d = d.view((-1, 512))
        
        out = torch.cat((d, t, m), dim = 1)
        
        out = self.fc(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.last_fc(out)
        
        # out = torch.softmax(self.last_fc(out), dim = 1)
        
        
        return out 
    


# Train and test functions
def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data_dir, data_time, data_meta, target) in enumerate(train_loader):
        data_dir = data_dir.float().to(device)
        data_time = data_time.float().to(device)
        data_meta = data_meta.float().to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data_dir, data_time, data_meta)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx%100:
            print ("Loss: {:0.6f}".format(loss.item()))
            
            
def test(model, device, loader):
    model.eval()
    correct = 0
    for batch_idx, (data_dir, data_time, data_meta, target) in enumerate(loader):
        data_dir = data_dir.float().to(device)
        data_time = data_time.float().to(device)
        data_meta = data_meta.float().to(device)
        target = target.to(device)
        
        output = model(data_dir, data_time, data_meta)
        output = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).float().sum().item()
    # print ('Accuracy Reports:\n Number of correct predictions: {}, length of dataset: {} accuracy: {:.3f}\n'. \
    #        format(correct, len(loader.dataset), correct / len(loader.dataset)))
    return correct / len(loader.dataset)



# Loading the data
num_mon_inst_test = 10
num_mon_inst_train = 80
num_mon_inst = num_mon_inst_test + num_mon_inst_train

train_dir = np.load(f'{data_path}/train_dir.npy')
train_time = np.load(f'{data_path}/train_time.npy')
train_metadata = np.load(f'{data_path}/train_metadata.npy')
train_labels = np.load(f'{data_path}/train_labels.npy')


test_dir = np.load(f'{data_path}/test_dir.npy')
test_time = np.load(f'{data_path}/test_time.npy')
test_metadata = np.load(f'{data_path}/test_metadata.npy')
test_labels = np.load(f'{data_path}/test_labels.npy')

# Convert lables from one-hot to classes
train_labels = np.argmax(train_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

print ("---------- Data dimensions")
print (f'Train data:')
print (f'directions: {train_dir.shape}, time: {train_time.shape}, metadata: {train_metadata.shape}, labels: {train_labels.shape}')

print (f'Test data:')
print (f'directions: {test_dir.shape}, time: {test_time.shape}, metadata: {test_metadata.shape}, labels: {test_labels.shape}')



# Wrapping data into data loaders
train_dataset = Data(train_dir, train_time, train_metadata, train_labels)
train_loader = DataLoader(train_dataset,
                          batch_size=50,
                          shuffle=True,
                          drop_last=True)

test_dataset = Data(test_dir, test_time, test_metadata, test_labels)
test_loader = DataLoader(test_dataset,
                        batch_size=batch_size,)



# Initializing model and optimizer
model = VarCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)



print ('-------- start training ...')
for epoch in range(num_epochs):
    print (f'----------------- Epoch {epoch} ------------------')
    train(model, device, train_loader, optimizer)
    # acc = test(model, device, valid_loader)
    # print (f'Validation accuracy: {acc*100:.2f}')
    
    
print ("====================================================================")

test_acc = test(model, device, test_loader)
print (f'Test accuracy: {acc*100:.2f}')
