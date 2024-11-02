import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT_MTL_Model(nn.Module):
    def __init__(self, in_features, hidden_dim, lstm_hidden_dim, out_features, num_heads=4):
        super(GAT_MTL_Model, self).__init__()

        # GAT Layers
        self.gat1 = GATConv(in_features, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=True)

        # LSTM layer to capture sequential dependencies after GAT
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)

        self.depression_head = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim, hidden_dim),  # 2 for bidirectional LSTM output
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim, hidden_dim),  # 2 for bidirectional LSTM output
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT layers
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.view(batch.size(0), -1, x.size(1))
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        # Task-specific outputs
        depression_output = torch.sigmoid(self.depression_head(lstm_out))
        severity_output = torch.softmax(self.severity_head(lstm_out), dim=1)

        return depression_output, severity_output
