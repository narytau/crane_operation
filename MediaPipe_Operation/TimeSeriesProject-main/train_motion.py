import os
import copy
import random
import numpy as np
import torch
from torch import nn, optim
import pickle
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.model.transformer import Transformer 
from ecg_dataset import myDataLoader


def create_data_array(data, frame_num, skip_num, is_stop):
    data_new = data[::skip_num, :]
    data_new = data_new.reshape(-1, data_new.shape[1] * frame_num)

    
    if is_stop:
        random_integers = [random.randint(0, data.shape[0]-1) for _ in range(int(data_new.shape[0] / 3))]
        data_stop = data[random_integers, :]
        data_stop = np.hstack([data_stop] * frame_num)
        data_new = np.vstack((data_new, data_stop))

    gauss_matrix = np.random.randn(data_new.shape[0], data_new.shape[1]) / 1000
    data_new += gauss_matrix

    return data_new 

def cross_entropy_loss(pred, target):

    criterion = nn.CrossEntropyLoss()
    #print('pred : '+ str(pred ) + ' target size: '+ str(target.size()) + 'target: '+ str(target )+   ' target2: '+ str(target))
    #print(  str(target.squeeze( -1)) )
    lossClass= criterion(pred, target ) 

    return lossClass


def calc_loss_and_score(pred, target, metrics): 
    softmax = nn.Softmax(dim=1)

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1)
    
    ce_loss = cross_entropy_loss(pred, target)
    #metrics['loss'] += ce_loss.data.cpu().numpy() * target.size(0)
    #metrics['loss'] += ce_loss.item()* target.size(0)
    metrics['loss'] .append( ce_loss.item() )
    pred = softmax(pred )
    
    #lossarr.append(ce_loss.item())
    #print('metrics : '+ str(ce_loss.item())  )
    #print('predicted max before = '+ str(pred))
    #pred = torch.sigmoid(pred)
    _,pred = torch.max(pred, dim=1)
    #print('predicted max = '+ str(pred ))
    #print('target = '+ str(target ))
    metrics['correct']  += torch.sum(pred ==target ).item()
    #print('correct sum =  '+ str(torch.sum(pred==target ).item()))
    metrics['total']  += target.size(0) 
    #print('target size  =  '+ str(target.size(0)) )

    return ce_loss


def print_metrics(main_metrics_train,main_metrics_val,metrics, phase):
    correct= metrics['correct']  
    total= metrics['total']  
    accuracy = 100*correct / total
    loss= metrics['loss'] 
    if(phase == 'train'):
        main_metrics_train['loss'].append( np.mean(loss)) 
        main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['loss'].append(np.mean(loss)) 
        main_metrics_val['accuracy'].append(accuracy ) 
    
    result = "phase: "+str(phase) \
    +  ' \nloss : {:4f}'.format(np.mean(loss))   +    ' accuracy : {:4f}'.format(accuracy)        +"\n"
    return result 


def train_model(dataloaders,model,optimizer, num_epochs=100): 

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_dict= dict()
    train_dict['loss']= list()
    train_dict['accuracy']= list() 
    val_dict= dict()
    val_dict['loss']= list()
    val_dict['accuracy']= list() 

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10) 

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = dict()
            metrics['loss']=list()
            metrics['correct']=0
            metrics['total']=0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                # zero the parameter gradients
                optimizer.zero_grad()
# 
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    #print('outputs size: '+ str(outputs.size()) )
                    loss = calc_loss_and_score(outputs, labels, metrics)   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print('epoch samples: '+ str(epoch_samples)) 
            print(print_metrics(main_metrics_train=train_dict, main_metrics_val=val_dict,metrics=metrics,phase=phase ))
            epoch_loss = np.mean(metrics['loss'])
        
            if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss 

    print('Best val loss: {:4f}'.format(best_loss))
    
def create_data_array(data, frame_num, skip_num, is_stop):
    data_new = data[::skip_num, :]
    data_new = data_new.reshape(-1, data_new.shape[1] * frame_num)

    
    if is_stop:
        random_integers = [random.randint(0, data.shape[0]-1) for _ in range(int(data_new.shape[0] / 3))]
        data_stop = data[random_integers, :]
        data_stop = np.hstack([data_stop] * frame_num)
        gauss_matrix = np.random.randn(data_stop.shape[0], data_stop.shape[1]) / 500
        data_stop += gauss_matrix
        
        data_new = np.vstack((data_new, data_stop))

    return data_new 

# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)

# HIGH, MIDDLE, LOW, STOP
MOTION_SPEED = [0, 1, 2]
frame_num = 30
feature = 6

# データを準備
scaler = StandardScaler()
data = np.loadtxt(os.path.join(CURRENT_PATH, "motion_data_original7.txt"))
# data_array = create_data_array(data, frame_num=frame_num, skip_num=1, is_stop=True)
data = scaler.fit_transform(data)

data_size = int(data.shape[0]/frame_num)
data_array = np.zeros((data_size, frame_num, feature))
for i in range(data_size):
    data_array[i,:,:] = data[frame_num*i:frame_num*(i+1)]

labels = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_array.shape[0] / len(MOTION_SPEED)))])

X_train, X_test, y_train, y_test = train_test_split(data_array, labels, test_size=0.15, random_state=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sequence_len = frame_num # sequence length of time series
max_len = 1000 # max time series sequence length 
n_head = 4 # number of attention head
n_layer = 2 # number of encoder layer
drop_prob = 0.1
d_model = 128 # number of dimension (for positional embedding)
ffn_hidden = 512 # size of hidden layer before classification 
feature = 6 # for univariate time series (1d), it must be adjusted for 1. 
batch_size = 32
model = Transformer(d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, feature=feature, details=False, device=device).to(device=device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
dataloaders = myDataLoader(batch_size=batch_size, x_train=X_train, x_val=X_test, y_train=y_train, y_val=y_test).getDataLoader()

model_normal_ce = train_model(dataloaders=dataloaders, model=model, optimizer=optimizer, num_epochs=20)
torch.save(model.state_dict(), 'myModel')

