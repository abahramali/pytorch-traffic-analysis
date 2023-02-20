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
num_classes = 95
num_epochs = 2
batch_size = 128
lr = 0.01


# functions to load the train and test data
def LoadTrainDataNoDefCW():
    print ("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '/work/abahramali_umass_edu/wf_datasets/deepwf/closed-world/'
    
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='latin1' ))
        print ('Train Loaded')
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='latin1'))
        
        
    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    
    
    return X_train, y_train


def LoadTestDataNoDefCW():
    dataset_dir = '/work/abahramali_umass_edu/wf_datasets/deepwf/closed-world/'
    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='latin1'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='latin1'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='latin1'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='latin1'))

    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)
    
    return X_valid, y_valid, X_test, y_test



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
        kernel_size = 8
        channels = [1, 32, 64, 128, 256]
        conv_stride = 1
        pool_stride = 4
        pool_size = 8
        
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size, stride = conv_stride)
        self.conv1_1 = nn.Conv1d(32, 32, kernel_size, stride = conv_stride)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size, stride = conv_stride)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, stride = conv_stride)
       
        self.conv3 = nn.Conv1d(64, 128, kernel_size, stride = conv_stride)
        self.conv3_3 = nn.Conv1d(128, 128, kernel_size, stride = conv_stride)
       
        self.conv4 = nn.Conv1d(128, 256, kernel_size, stride = conv_stride)
        self.conv4_4 = nn.Conv1d(256, 256, kernel_size, stride = conv_stride)
       
        
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(256)
        
        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        

        self.fc1 = nn.Linear(3328, 512)
        self.batch_norm_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.last_fc = nn.Linear(512, num_classes)
        self.dropout_fc1 = nn.Dropout2d(p=0.7)
        self.dropout_fc2 = nn.Dropout2d(p=0.5)
        
        self.dropout_conv = nn.Dropout2d(p=0.1)
        
        

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()
            
        
    def forward(self, inp):
        x = inp
        x = F.elu((self.conv1(x)))
        x = F.elu(self.batch_norm1(self.conv1_1(x)))
        x = self.max_pool_1(x)
        x = self.dropout_conv(x)
        
        x = F.relu((self.conv2(x)))
        x = F.relu(self.batch_norm2(self.conv2_2(x)))
        x = self.max_pool_2(x)
        x = self.dropout_conv(x)
       
        x = F.relu((self.conv3(x)))
        x = F.relu(self.batch_norm3(self.conv3_3(x)))
        x = self.max_pool_3(x)
        x = self.dropout_conv(x)
      
        x = F.relu((self.conv4(x)))
        x = F.relu(self.batch_norm4(self.conv4_4(x)))
        x = self.max_pool_4(x)
        x = self.dropout_conv(x)
                
        x = x.view(x.size(0), -1)
        
        x = F.relu((self.fc1(x)))
        x = self.dropout_fc1(x)
        x = F.relu(self.batch_norm_fc(self.fc2(x)))
        x = self.dropout_fc2(x)
        x = self.last_fc(x)
        
        return x    
    
    
    
# train and test functions
def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), 1, data.size(1)).float().to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%100 == 0:
            print ("Loss: {:0.6f}".format(loss.item()))
        

def test(model, device, loader):
    model.eval()
    correct = 0
    temp = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.size(0), 1, data.size(1)).float().to(device)
            target = target.to(device)
            
            output = model(data)
            output = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).float().sum().item()
    # print ('Accuracy Reports:\n Number of correct predictions: {}, length of dataset: {} accuracy: {:.3f}\n'. \
    #        format(correct, len(loader.dataset), correct / len(loader.dataset)))
    return correct / len(loader.dataset)




# Load the data and wrap them using the data loaders
x_train, y_train = LoadTrainDataNoDefCW()
x_valid, y_valid, x_test, y_test = LoadTestDataNoDefCW()

train_dataset = Data(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

valid_dataset = Data(x_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)

test_dataset = Data(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)



# initialize the model and optimizer
model = Net().to(device)
model.weight_init()
optimizer = optim.Adam(model.parameters(),lr = lr)



print ('-------- start training ...')
for epoch in range(num_epochs):
    print (f'----------------- Epoch {epoch} ------------------')
    train(model, device, train_loader, optimizer)
    acc = test(model, device, valid_loader)
    print (f'Validation accuracy: {acc*100:.2f}')
    
    
print ("====================================================================")

test_acc = test(model, device, test_loader)
print (f'Test accuracy: {acc*100:.2f}')



