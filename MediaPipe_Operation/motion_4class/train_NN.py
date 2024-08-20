import os
import torch
import pickle
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

from class_NN import RegularizedNN, ComplexNN, TransformerModel

# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(CURRENT_PATH)
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "gesture_classifier6.pth")

MOTION_SPEED = [0, 1, 2, 3]
is_saved = False
frame_num = 25

scaler = pickle.load(open(os.path.join(SAVE_PATH, 'motion_scaler6.sav'), 'rb'))

# Data preparation
data_array = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data6.txt'))

print(data_array.shape)

# データの標準化
data_array = scaler.fit_transform(data_array)

EACH_DATA = 800
label_array = np.array([speed for speed in MOTION_SPEED for _ in range(int(data_array.shape[0] / len(MOTION_SPEED)))])
# label_array = np.hstack(([0]*EACH_DATA, [1]*EACH_DATA, [2]*2*EACH_DATA, [3]*2*EACH_DATA))


X_train, X_test, y_train, y_test = train_test_split(data_array, label_array, train_size=0.75, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

input_size = 6 * frame_num
num_classes = len(MOTION_SPEED)



######################################################
# layer = [10*_ for _ in range(1,9)]
# drop = [_*0.1 for _ in range(1,10)]
# avg = []
# for i in range(75,85):
#     for j in range(35,45):
#         # for k in range(j):
        
#         num1, num2= i,j
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
model = ComplexNN(input_size, num_classes, 76, 43)


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

print("Val accuracy average:")
print(sum(val_array[-10:]) / len(val_array[-10:]))


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
print(accuracy_score(all_labels, all_predictions))