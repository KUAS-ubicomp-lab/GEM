import torch.nn as nn
from torch_geometric.nn import GATConv
from transformers import AutoModel, AutoTokenizer


class GAT_MTL_Model(nn.Module):
    def __init__(self, in_channels, hidden_channels, lstm_hidden_size, num_classes_binary, num_classes_severity, num_heads=4):
        super(GAT_MTL_Model, self).__init__()

        self.num_heads = num_heads
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.hidden_size = self.bert_model.config.hidden_size

        # Hierarchical GAT with different edge types (root-sub, sub-sub)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True, residual=True)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=4, concat=False, residual=True)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(hidden_channels, lstm_hidden_size, batch_first=True, bidirectional=True)

        # Shared layer for severity and depression
        self.fc_shared = nn.Linear(hidden_channels, hidden_channels)

        self.binary_fc = nn.Linear(2 * lstm_hidden_size, num_classes_binary)
        self.severity_fc = nn.Linear(2 * lstm_hidden_size, num_classes_severity)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        x = self.relu(self.fc_shared(x))
        x = self.dropout(self.fc_shared(x))

        out_depression = self.binary_fc(x)
        out_severity = self.severity_fc(x)

        return out_depression, out_severity
