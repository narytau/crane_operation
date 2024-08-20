import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# Generate data
# Constants and Paths
CURRENT_PATH = os.path.dirname(__file__)
BASE_PATH = os.path.dirname(CURRENT_PATH)
SAVE_PATH = os.path.join(BASE_PATH, "model_4class")
DATA_SAVE_PATH = os.path.join(BASE_PATH, "data_4class")

MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "gesture_classifier_transformer.pth")

MOTION_SPEED = [0, 1, 2]

data = np.loadtxt(os.path.join(DATA_SAVE_PATH, 'motion_data_transformer.txt'))
labels = np.array([speed for speed in MOTION_SPEED for _ in range(int(data.shape[0] / len(MOTION_SPEED)))])

# Parameters
num_samples, num_features = data.shape[0], data.shape[1]

# Shuffle the data
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Plot an example from each class
# plt.figure(figsize=(15, 5))
# for i, label in enumerate(['A', 'B', 'C']):
#     plt.subplot(1, 3, i+1)
#     plt.plot(data[labels == i][:100])
#     plt.title(f"Class {label}")
# plt.show()


# Define Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        output = self.fc(output)
        return output

# Parameters
print(num_features)
input_dim = num_features
nhead = 2
num_encoder_layers = 3
dim_feedforward = 128
output_dim = 3
batch_size = 64
epochs = 30
learning_rate = 0.001

# Prepare data
tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Add batch dimension
tensor_labels = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(tensor_data, tensor_labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = TransformerModel(input_dim, nhead, num_encoder_layers, dim_feedforward, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.permute(1, 0, 2)  # (seq_length, batch_size, num_features)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        inputs = inputs.permute(1, 0, 2)  # (seq_length, batch_size, num_features)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

import time

def classify_real_time_data(model, data_stream):
    model.eval()
    buffer = []

    for data_point in data_stream:
        buffer.append(data_point)
        input_sequence = torch.tensor(buffer, dtype=torch.float32).unsqueeze(1)  # (seq_length, 1, num_features)
        with torch.no_grad():
            output = model(input_sequence)
            _, predicted = torch.max(output, 1)
            print(f"Predicted class: {predicted.item()}")
        time.sleep(0.1)  # Simulate real-time data arrival


# Save the model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Simulate real-time data stream with new data points
# real_time_data_stream = np.random.randn(1000, num_features) * 5  # Change this to match the pattern you want to test
# classify_real_time_data(model, real_time_data_stream)
