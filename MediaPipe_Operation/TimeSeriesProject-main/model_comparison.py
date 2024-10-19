import os
import time
import torch
import pickle
import numpy as np
from class_NN import ComplexNN, ComplexNN2, RegularizedNN
from models.model.transformer import Transformer 


# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH    = os.path.dirname(CURRENT_PATH)
TASK_PATH = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
HAND_TASK_PATH = os.path.join(BASE_PATH, "recognizer", "gesture_recognizer.task")
MODEL_PATH = os.path.join(BASE_PATH, "motion_previous_model", "theta_data")
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")

print(SAVE_PATH)
frame_num = 30
input_size = 6 * frame_num  # 入力ベクトルの長さ
num_classes = 4             # クラスの数

# model_1 SVM
model_1 = pickle.load(open(os.path.join(SAVE_PATH, 'model_SVM_slide.sav'), 'rb'))

# model_2 Neural Network
# model_2 = ComplexNN(input_size, num_classes, num1=126, num2=75)
model_2 = RegularizedNN(input_size, num_classes, 50)

model_2.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'gesture_classifier_with_unclass_data2.pth')))
model_2.eval()

# model_3 Transformer

# definition of model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sequence_len = frame_num # sequence length of time series
max_len = 1000 # max time series sequence length 
n_head = 4 # number of attention head
n_layer = 3 # number of encoder layer
drop_prob = 0.1
d_model = 128 # number of dimension (for positional embedding)
ffn_hidden = 512 # size of hidden layer before classification 
feature = 6 # for univariate time series (1d), it must be adjusted for 1. 
batch_size = 1

model_3 = Transformer(d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, feature=feature, details=False, device=device).to(device=device)
model_3.load_state_dict(torch.load('myModel'))
model_3.eval()  # 評価モードに設定

# Create dummy test data (replace with actual data)
DATA_NUM = 1000
X_test = torch.rand(DATA_NUM, input_size)  # Example for an image model

# model_1
total_time = 0
for i in range(DATA_NUM):
    start_time = time.time()
    model_1.predict(X_test[i, :].reshape(1, input_size))
    end_time = time.time()
    
    total_time += end_time - start_time

# Measure time for PyTorch models
print(f"Model 1 time: {total_time:.4f} seconds")

# model 2
total_time = 0
for i in range(DATA_NUM):
    input_tensor = torch.tensor(X_test[i, :], dtype=torch.float32)
    # input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        start_time = time.time()
        output = model_2(input_tensor)
        end_time = time.time()

    total_time += end_time - start_time

# Measure time for PyTorch models
print(f"Model 2 time: {total_time:.4f} seconds")


# model 3
total_time = 0
for i in range(DATA_NUM):
    array_tensor = np.zeros((1, frame_num, 1, 6))
    array_tensor[0, :, 0, :] = X_test[i, :].reshape(frame_num, 6)
    array_tensor = torch.tensor(array_tensor, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        output = model_3(array_tensor)
        end_time = time.time()
        
    total_time += end_time - start_time

print(f"Model 3 time: {total_time:.4f} seconds")
