import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_hidden_layer_1: int = 64,
        n_hidden_layer_2: int = 32,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden_layer_1)
        self.fc2 = nn.Linear(n_hidden_layer_1, n_hidden_layer_2)
        self.fc3 = nn.Linear(n_hidden_layer_2, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        y_hat = self.fc3(x)
        return y_hat


class LogRegression(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(LogRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.linear(x)
        y_pred = torch.sigmoid(x)
        return y_pred


class SVM(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
