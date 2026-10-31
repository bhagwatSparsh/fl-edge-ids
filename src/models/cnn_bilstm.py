"""
CNN-BiLSTM (plan Phase 1): input reshaped to (F, 1) -> Conv1d(1->32, k=3) ->
ReLU -> Conv1d(32->64, k=3) -> ReLU -> MaxPool -> BiLSTM(hidden 64) ->
Dense(64) -> C.

Per the stack review (method issue #10): this treats the 115/60-feature axis
as a sequence, which is a stated modeling choice (not literal time), and the
MLP baseline exists specifically to check whether this earns its extra
complexity.
"""
import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True,
                             bidirectional=True)
        self.dense = nn.Linear(64 * 2, 64)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, n_features) -> (batch, 1, n_features)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)                       # (batch, 64, n_features//2)
        x = x.transpose(1, 2)                  # (batch, n_features//2, 64) -- seq of 64-dim steps
        out, _ = self.lstm(x)                  # (batch, seq, 128)
        last = out[:, -1, :]                   # final timestep, both directions
        h = self.relu(self.dense(last))
        return self.out(h)
