import os
import random
import numpy as np
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RealTimeTimeSeriesDataset(Dataset):
    def __init__(self, data, labels, window_size=30):
        self.data = data
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.labels[idx + self.window_size - 1]  # ウィンドウの最後のフレームのラベルを使用
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

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
TASK_PATH    = os.path.join(BASE_PATH, "recognizer", "pose_landmarker_full.task")
MODEL_PATH   = os.path.join(BASE_PATH, "theta_data")
SAVE_PATH    = os.path.join(BASE_PATH, "motion_model_transformer")

# HIGH, MIDDLE, LOW, STOP
MOTION_SPEED = [0, 1, 2, 3]

# データを準備
scaler = StandardScaler()
data = np.loadtxt(os.path.join(SAVE_PATH, 'motion_data_original3.txt'))
data = data[::2, :]
data = scaler.fit_transform(data)

labels = np.array([speed for speed in MOTION_SPEED for _ in range(int(data.shape[0] / len(MOTION_SPEED)))])

X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0)
print(X_train.shape, X_test.shape)

window_size = 50
dataset = RealTimeTimeSeriesDataset(X_train, y_train, window_size=window_size)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_length=30):
        super(TimeSeriesTransformer, self).__init__()
        config = BertConfig(
            vocab_size=1,  # テキストデータではないため、ダミーのvocab_size
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=seq_length,  # 30の長さのシーケンス
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.linear = nn.Linear(input_dim, config.hidden_size)
        self.transformer = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, num_classes)

    # def forward(self, x):
    #     batch_size, seq_length, input_dim = x.shape
    #     x = x.view(batch_size * seq_length, input_dim)  # 変形 (batch_size*seq_length, input_dim)
    #     print(x.shape)
    #     x = x.unsqueeze(1)  # (batch_size, seq_length, input_dim) -> (batch_size, seq_length, 1, input_dim)
    #     x = x.transpose(1, 2)  # (batch_size, 1, seq_length, input_dim)
    #     transformer_output = self.transformer(inputs_embeds=x).last_hidden_state
    #     transformer_output = transformer_output.view(batch_size, seq_length, -1)  # 元の形状に戻す
    #     transformer_output = transformer_output[:, 0, :]  # (batch_size, hidden_size)
    #     output = self.fc(transformer_output)  # (batch_size, num_classes)
    #     return output
    
    def forward(self, x):
        # batch_size, seq_length, input_dim = x.shape
        # x = x.view(batch_size, seq_length, input_dim)  # (batch_size, seq_length, input_dim)
        # x = self.transformer(inputs_embeds=x).last_hidden_state  # (batch_size, seq_length, hidden_size)
        x = self.linear(x)
        x = self.transformer(inputs_embeds=x).last_hidden_state
        x = x.mean(dim=1)  # (batch_size, hidden_size) 時系列データ全体の平均をとる
        output = self.fc(x)  # (batch_size, num_classes)
        return output

# モデルの初期化
model = TimeSeriesTransformer(input_dim=6, num_classes=4, seq_length=window_size)  # データに応じてクラス数を設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

train_model(model, train_loader, criterion, optimizer)

# モデルの保存
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "transformer_model.pth")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

pickle.dump(scaler, open(os.path.join(SAVE_PATH, 'motion_scaler.sav'), 'wb'))


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {int(100 * correct / total)}%')

# テストデータの用意と評価
test_data = X_test
test_labels = y_test
test_dataset = RealTimeTimeSeriesDataset(test_data, test_labels, window_size=30)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
evaluate_model(model, test_loader)
