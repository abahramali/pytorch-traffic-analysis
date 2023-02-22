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
import collections
import random
import warnings
warnings.filterwarnings("ignore") 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# find the available gpu device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
print ("Available cuda device:", device)


# Parameters for training
num_classes = 100
num_epochs = 2
batch_size = 128
alpha = 0.1
alpha_value = float(alpha)
num_ins_per_website = 25
momentum = 0.9
weight_decay = 1e-6
lr = 0.001
embedding_size = 64
test_split = 0.15
val_split = 0.15
data_path = '/work/abahramali_umass_edu/wf_datasets/awf/tor_100w_2500tr.npz'



# Triplet mining
def build_similarities(conv, all_imgs):
    embs = conv(all_imgs)
    embs = embs.detach().cpu().data.numpy()
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    return all_sims


def intersect(a, b):
    return list(set(a) & set(b))

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
    if similarities is None:
        return random.sample(neg_imgs_idx,len(anc_idxs))
    final_neg = []
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        sim = similarities[anc_idx, pos_idx]
        possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg


class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, conv):
        self.batch_size = batch_size

        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.all_anchors = list(set(Xa_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.neg_traces_idx)}
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0
                break
            
            traces_a = self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)

            yield ([self.traces[traces_a],
                    self.traces[traces_p],
                    self.traces[traces_n]],
                   np.zeros(shape=(traces_a.shape[0]))
                   )
            

# Loss functions
class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, inp):
        positive_sim, negative_sim = inp
#         print (negative_sim, positive_sim)
        losses = torch.max(torch.tensor(0.0).to(device), negative_sim - positive_sim + self.alpha)
        return torch.mean(losses)
        
        
class IdentityLoss(nn.Module):
    def forward(self, y_true, y_pred):
        return torch.mean(y_pred, - 0 * y_true)
    
    
    
# Model
class DFNet(nn.Module):
    def __init__(self, embedding_size):
        super(DFNet, self).__init__()
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
       
                
        self.max_pool_1 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        
        self.drop_out = nn.Dropout(p=0.1)
        self.drop_out2 = nn.Dropout()

        self.fc1 = nn.Linear(4608, 512)
        self.batch_norm_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.last_fc = nn.Linear(512, 64)
        self.dropout_fc1 = nn.Dropout(p=0.7)
        self.dropout_fc2 = nn.Dropout(p=0.5)
        
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.1)
        self.dropout3 = nn.Dropout2d(p=0.1)
        self.dropout4 = nn.Dropout2d(p=0.1)
        
        self.tf_fc = nn.Linear(4608, embedding_size)

    def weight_init(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
#                 m.weight.data.xavier_uniform_()
                print (n)
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()
            
        
    def forward(self, inp):
        x = inp
        x = F.pad(x, (3,4))
        x = F.elu((self.conv1(x)))
        x = F.pad(x, (3,4))
        x = F.elu(self.conv1_1(x))
        x = self.max_pool_1(x)
        x = self.dropout1(x)
        
        x = F.pad(x, (3,4))
        x = F.relu((self.conv2(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.conv2_2(x))
        x = self.max_pool_2(x)
        x = self.dropout2(x)
       
        x = F.pad(x, (3,4))
        x = F.relu((self.conv3(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.conv3_3(x))
        x = self.max_pool_3(x)
        x = self.dropout3(x)
      
        x = F.pad(x, (3,4))
        x = F.relu((self.conv4(x)))
        x = F.pad(x, (3,4))
        x = F.relu(self.conv4_4(x))
        x = self.max_pool_4(x)
                
        x = x.view(x.size(0), -1)
        
        x = self.tf_fc(x)
        
        
        return x    
        
        
# Triplet model
class ModelTriplet(nn.Module):
    def __init__(self, shared_df):
        super(ModelTriplet, self).__init__()
        self.shared_df = shared_df
        
        self.triplet_loss = TripletLoss(alpha_value)
        
    def forward(self, inp):
        anchor, positive, negative = inp
        
        a = self.shared_df(anchor)
        p = self.shared_df(positive)
        n = self.shared_df(negative)
        
        pos_sim = F.cosine_similarity(a, p, dim = -1)
        neg_sim = F.cosine_similarity(a, n, dim = -1)
        
        triplet_inp = (pos_sim, neg_sim)
        
        return self.triplet_loss(triplet_inp)
    
    
    
    
# Train and test functions
def train(model, device, optimizer, generator, num_iterations):
    model.train()
    for idx, (inp, _) in enumerate(generator.next_train()):
        anchor, positive, negative = inp    
        
        optimizer.zero_grad()
        output = model((anchor, positive, negative))
        
        loss = torch.mean(output)
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            print ('Iteration:', idx, ' loss:', loss)

# Loading the AWF data
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
    
    npzfile = np.load(data_path, allow_pickle=True)
    data = npzfile["data"]
    labels = npzfile["labels"]
    npzfile.close()
    
    # data = data.reshape(data.shape[0], 1, data.shape[1])
    
    labels = categorize(labels)
    
    return np.array(data), np.array(labels)

# Function to sample N samples from each website
def sample_traces(x, y, N):
    train_index = []
    
    for c in range(num_classes-1):
        idx = np.where(y == c)[0]
        idx = np.random.choice(idx, min(N, len(idx)), False)
        train_index.extend(idx)
        
    train_index = np.array(train_index)
    np.random.shuffle(train_index)
    
    x_train = x[train_index]
    y_train = y[train_index]
    
    return x_train, y_train

def build_pos_pairs_for_id(classid):
    traces = classid_to_ids[classid]
    
    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs

def build_positive_pairs(class_id_range):
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id)
        
        for pair in pos:
            listX1 += [pair[0]]
            listX2 += [pair[1]]
    perm = np.random.permutation(len(listX1))
    
    return np.array(listX1)[perm], np.array(listX2)[perm]



data, target = load_data()

num_instances = data.shape[0]
num_cells = data.shape[1]
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
x_valid, y_valid = data[ind_val], target[ind_val]
x_test, y_test = data[ind_test], target[ind_test]


del data, target

# print ("Data dimensions:")
# print (f'Train data: {x_train.shape}, {y_train.shape}')
# print (f'Validation data: {x_valid.shape}, {y_valid.shape}')
# print (f'Test data: {x_test.shape}, {y_test.shape}')



# Randomly sampling `num_ins_per_website` from each class
x_train, y_train = sample_traces(x_train, y_train, num_ins_per_website)

classid_to_ids = collections.defaultdict(list)
for idx in range(len(x_train)):
    classid_to_ids[y_train[idx]].append(idx)
    

all_traces = x_train
all_traces = all_traces[:, np.newaxis, :]

print ("Load traces with ",all_traces.shape)
print ("Total size allocated on RAM : ", str(all_traces.nbytes / 1e6) + ' MB')


id_to_classid = {v:c for c, traces in classid_to_ids.items() for v in traces}


Xa_train, Xp_train = build_positive_pairs(range(num_classes))

all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
print ('Feature extractor data dimensions:')
print ("X_train Anchor: ", Xa_train.shape)
print ("X_train Positive: ", Xp_train.shape)
            
            
# Convert all traces to tensors
all_traces = torch.from_numpy(all_traces).float().to(device)



# Initializing the model and optimizer
shared_model = DFNet(embedding_size).to(device)
triplet_model = ModelTriplet(shared_model)
optimizer = optim.SGD(triplet_model.parameters(),
                      lr=lr,
                      momentum=momentum,
                      weight_decay=weight_decay,
                      nesterov=True)

num_iterations = Xa_train.shape[0] // batch_size
print (f'Number of iterations: {num_iterations}')



# Initializing the generator
gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None)    



# Training
print ('-------- start training ...')
# for epoch in range(num_epochs):
#     print ("built new hard generator for epoch " + str(epoch))
#     train(triplet_model, device, optimizer, gen_hard, num_iterations)
#     gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, shared_model)

    
    
    
    
# ---------------------------------- KNN classifier for testing
N = 5 # Number of labeled samples for training the KNN classifier 

# Function to convert traces to embeddings
def convert_to_embeddings(data):
    embeddings = []
    
    for x in data:
        x = x.reshape((1,1,5000))
        x = torch.from_numpy(x).float().to(device)
        out = shared_model(x)
        out = out.cpu().data.numpy()
        embeddings.append(out)
        
    return np.array(embeddings).reshape((-1, 64))


print ('---------------------------- KNN classifier')
# split the test data into train and test for KNN classifier
knn_train_indices = []
knn_test_indices = []

for c in range(num_classes):
    indices = np.where(y_test == c)[0]
    np.random.shuffle(indices)
    
    knn_train_indices.extend(indices[:N]) # N samples per website to train the KNN classifier
    knn_test_indices.extend(indices[N:N+100]) # 100 samples per website to test the KNN classifier
    
x_train_knn = x_test[knn_train_indices]
y_train_knn = y_test[knn_train_indices]

x_test_knn = x_test[knn_test_indices]
y_test_knn = y_test[knn_test_indices]


# Convert traces to embeddings
x_train_embeddings = convert_to_embeddings(x_train_knn)
x_test_embeddings = convert_to_embeddings(x_test_knn)



# Initializing KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2, metric='cosine', algorithm='brute')
knn.fit(x_train_embeddings, y_train_knn)

acc_knn_top1 = accuracy_score(y_test_knn, knn.predict(x_test_embeddings))

print (f'KNN accuracy on with {N} labeled samples per website: {acc_knn_top1:.2f}')