import os
import cv2
import time
import pickle
import random
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# Model definition
class RegularizedNN(nn.Module):
    def __init__(self, input_size, num_classes, num):
        super(RegularizedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, num)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(num, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

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

    
# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

MOTION_SPEED = [0, 1, 2, 3]

data_array = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_original4.txt'))
print(data_array.shape)
############ Adjust the num of arrays #################
# len1 = data_array3.shape[0]
# random_numbers1 = random.sample(range(0, data_array1.shape[0]), len1)
# random_numbers2 = random.sample(range(0, data_array2.shape[0]), len1)

# data_array1 = data_array1[random_numbers1, :]
# data_array2 = data_array2[random_numbers2, :]

# label_array1 = label_array1[random_numbers1]
# label_array2 = label_array2[random_numbers2]
#######################################################


frame_num_array = [15, 20, 25]


score_array = np.zeros((len(frame_num_array), 4))


# # frame_num
# for i in range(len(frame_num_array)):
#     # how many data we should take
#     for j in range(1,5):
#         # data = data_array[i][::j, :]
        
#         # data = scaler.fit_transform(data)
#         # label = label_array[i][::j]
#         data = create_data_array(data_array, frame_num=frame_num_array[i], skip_num=j, is_stop=False)
#         label = np.array([speed for speed in MOTION_SPEED for _ in range(int(data.shape[0] / len(MOTION_SPEED)))])
        
#         print(i, j)
#         # random_numbers = random.sample(range(0, data.shape[0]), 1280)

#         # data = data[random_numbers, :]
#         # label = label[random_numbers]

#         scaler = StandardScaler()
#         data = scaler.fit_transform(data)


#         X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0)
        
#         train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
#         test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        
#         train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#         input_size = 6 * frame_num_array[i]
#         num_classes = len(MOTION_SPEED)
#         model = RegularizedNN(input_size, num_classes, 70)
        
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.025, weight_decay=0.01)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

#         # Early stopping parameters
#         patience = 20
#         best_loss = float('inf')
#         best_model = None
#         counter = 0

#         val_array = []

#         # Training loop
#         num_epochs = 1000
#         for epoch in range(num_epochs):
#             model.train()
#             train_loss = 0.0
#             for inputs, labels in train_loader:
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()

#             train_loss /= len(train_loader)

#             # Validation loop
#             model.eval()
#             val_loss = 0.0
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for inputs, labels in test_loader:
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     val_loss += loss.item()
#                     _, predicted = torch.max(outputs, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()

#             val_loss /= len(test_loader)
#             val_accuracy = correct / total
            
#             val_array.append(val_accuracy)

#             scheduler.step(val_loss)

#             print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

#             # Early stopping
#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 best_model = model.state_dict()
#                 counter = 0
#             else:
#                 counter += 1

#             if counter >= patience:
#                 print("Early stopping")
#                 break

#         print("Val accuracy average:")
#         print(sum(val_array[-10:]) / len(val_array[-10:]))

#         # Get predictions
#         all_labels = []
#         all_predictions = []
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 all_labels.extend(labels.cpu().numpy())
#                 all_predictions.extend(predicted.cpu().numpy())

#         score_array[i, j-1] = accuracy_score(all_labels, all_predictions)


# print(score_array)

#################################################################################

# frame_num
for i in range(len(frame_num_array)):
    # how many data we should take
    for j in range(1,4):
        data = create_data_array(data_array, frame_num=frame_num_array[i], skip_num=j, is_stop=False)
        label = np.array([speed for speed in MOTION_SPEED for _ in range(int(data.shape[0] / len(MOTION_SPEED)))])
        
        # random_numbers = random.sample(range(0, data.shape[0]), 1280)

        # data = data[random_numbers, :]
        # label = label[random_numbers]

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        
        X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0)
        
        print(data.shape)
        
        model = SVC()
        # param_grid = {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
        # grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        score_array[i, j-1] = score

print(score_array)        

"""
SVM (Default)  --> frame 25 / every frame is the best
[[0.790625   0.74875    0.76217228]
 [0.83       0.79833333 0.775     ]
 [0.84583333 0.78333333 0.8375    ]]

SVM (Default) Same data volume
[[0.740625 0.709375 0.70625 ]
 [0.778125 0.78125  0.75625 ]
 [0.790625 0.859375 0.796875]]
 
NN (1 hidden layer (60))
[[0.853125 0.85     0.88125 ]
 [0.85625  0.875    0.903125]
 [0.928125 0.903125 0.90625 ]]
 
NN (1 hidden layer (60)) Same data volume
 [[0.825    0.83125  0.815625]
 [0.925    0.890625 0.890625]
 [0.921875 0.915625 0.896875]]
 
NN (1 hidden layer (50)) Same data volume
 [[0.85     0.884375 0.821875]
 [0.88125  0.89375  0.896875]
 [0.9125   0.89375  0.934375]]
 
 
 not same
 [[0.9075     0.87625    0.88389513 0.8825    ]
 [0.93583333 0.905      0.8625     0.82      ]
 [0.940625   0.73125    0.940625   0.90416667]]
"""
