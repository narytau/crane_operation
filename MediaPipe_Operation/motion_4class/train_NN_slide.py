import os
import sys
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from class_NN import RegularizedNN, ComplexNN, TransformerModel, ComplexNN2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from my_module.sliding_time_window import sliding_time_window

def apply_augmentations(data):
    # Example augmentation: Time warp
    time_warp_ratio = 0.05
    warp_factor = np.random.uniform(1 - time_warp_ratio, 1 + time_warp_ratio, data.shape[0])
    print(warp_factor)
    for i in range(data.shape[0]):
        data[i] = data[i] * warp_factor[i]
        
    # Add Gaussian noise
    gauss_matrix = np.random.randn(data.shape[0], data.shape[1]) / 1000
    data += gauss_matrix
    return data
        
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

    data_new = apply_augmentations(data_new)

    return data_new 


# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(CURRENT_PATH)
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "gesture_classifier_with_unclass_data2.pth")

MOTION_SPEED = [0, 1, 2, 3]
is_saved = True
frame_num = 30

# Data preparation
data_array = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_original10.txt'))
data_unclass1 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass10.txt'))
data_unclass2 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass11.txt'))
data_unclass3 = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_unclass12.txt'))
# data_unclass = np.vstack((data_unclass1.reshape(-1, data_unclass1.shape[1]*frame_num),
#                         data_unclass2.reshape(-1, data_unclass2.shape[1]*frame_num),
#                         data_unclass3.reshape(-1, data_unclass3.shape[1]*frame_num)))

data_unclass = np.vstack((data_unclass2, data_unclass3))


random_integers = [random.randint(0, data_array.shape[0]-1)  for _ in range(int(data_array.shape[0] / 3 / frame_num))]
random_integers = [i for i in random_integers for _ in range(frame_num)]

data_stop = data_array[random_integers, :]

data_array_new = np.vstack((data_array, data_stop))
data_array_new[-data_unclass.shape[0]:, :] = data_unclass

gauss_matrix = np.random.randn(data_array_new.shape[0], data_array_new.shape[1]) / 500
data_array_new += gauss_matrix
# data_array_new = data_array_new[::2, :]


# sliding time window
data_each_num = int(data_array_new.shape[0] / 4)
for i in range(4):
    data_each = data_array_new[i*data_each_num:(i+1)*data_each_num, :]
    data_window_each = sliding_time_window(array=data_each, window_size=frame_num, step_size=20)
    if i == 0:
        data_window_array = data_window_each
    else:
        data_window_array = np.concatenate([data_window_array, data_window_each], axis=0)

# Reshape: data_num x frame_num x features --> data_num x (frame_num x features)
data_window_array = np.reshape(np.transpose(data_window_array, (0,2,1)), 
                            (data_window_array.shape[0],-1))


# データの標準化
scaler = StandardScaler()
data_array = scaler.fit_transform(data_array)

# pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler_with_unclass_data2.sav'), 'wb'))


label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_window_array.shape[0] / len(MOTION_SPEED)))])

X_train, X_test, y_train, y_test = train_test_split(data_window_array, label_array, train_size=0.75, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

input_size = 6 * frame_num
num_classes = len(MOTION_SPEED)


######################################################
# layer = [10*_ for _ in range(1,15)]
# drop = [_*0.1 for _ in range(1,10)]
# avg = []
# for i in range(len(layer)):
#     for j in range(i):
#         # for k in range(j):
        
#         # num1, num2= i,j
#         num1, num2= layer[i], layer[j]

#         print(num1, num2)
#         # model = TransformerModel(input_size, num_classes)
#         # model = RegularizedNN(input_size, num_classes, num1)
#         model = ComplexNN(input_size, num_classes, num1, num2)


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
#                 torch.save(model.state_dict(), MODEL_SAVE_PATH)
#             else:
#                 counter += 1

#             if counter >= patience:
#                 print("Early stopping")
#                 break

#         avg.append(sum(val_array[-10:]) / len(val_array[-10:]))

# print("Val accuracy average:")
# print(layer)
# print(drop)
# print(avg)
# print(max(avg))

##########################################################################

# model = TransformerModel(input_size, num_classes)
# model = RegularizedNN(input_size, num_classes, 60)
model = ComplexNN(input_size, num_classes, 126, 75)
# model = ComplexNN2(input_size, num_classes, 130, 80, 50)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.025, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# Early stopping parameters
patience = 20
best_loss = float('inf')
best_model = None
counter = 0

val_array = []

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader)
    val_accuracy = correct / total
    
    val_array.append(val_accuracy)

    scheduler.step(val_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model.state_dict()
        counter = 0
        # torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        break

# Save the final model
if is_saved:
    if best_model is not None:
        model.load_state_dict(best_model)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)



# Get predictions
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate precision, recall, F1-score, and support
report = classification_report(all_labels, all_predictions, target_names=[str(cls) for cls in MOTION_SPEED])
print(report)
print("accuracy_score")
print(accuracy_score(all_labels, all_predictions))