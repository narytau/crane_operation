import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
    
class ComplexNN(nn.Module):
    def __init__(self, input_size, num_classes, num1, num2):
        # num1, num2 = 82, 40
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, num1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(num1, num2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(num2, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class ComplexNN2(nn.Module):
    def __init__(self, input_size, num_classes,num1, num2, num3):
        # num1, num2 = 82, 40
        super(ComplexNN2, self).__init__()
        self.fc1 = nn.Linear(input_size, num1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(num1, num2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(num2, num3)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(num3, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class ComplexNN3(nn.Module):
    def __init__(self, input_size, num_classes,num1, num2, num3):
        # num1, num2 = 82, 40
        super(ComplexNN2, self).__init__()
        self.fc1 = nn.Linear(input_size, num1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(num1, num2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(num2, num3)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(num3, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, num_heads=2, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, 64)
        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence length dimension
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, embedding_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.fc(x)
        return x
