import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from transformers import AutoModel


class GAT_MTL_Model(nn.Module):
    def __init__(self, hidden_channels, lstm_hidden_size, num_classes_binary, num_classes_severity, num_heads=2):
        super(GAT_MTL_Model, self).__init__()

        self.num_heads = num_heads
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        self.hidden_size = self.bert_model.config.hidden_size

        # Hierarchical GAT with different edge types (root-sub, sub-sub)
        self.gat1 = GATConv(self.hidden_size + hidden_channels, hidden_channels, num_relations=2, residual=True)
        self.gat2 = GATConv(hidden_channels, hidden_channels, num_relations=2, residual=True)

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(hidden_channels, lstm_hidden_size, batch_first=True, bidirectional=True)

        self.binary_fc = nn.Linear(2 * lstm_hidden_size, num_classes_binary)
        self.severity_fc = nn.Linear(2 * lstm_hidden_size, num_classes_severity)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        # Obtain discourse-aware embeddings with BERT
        bert_output = self.bert_model(data.input_ids, attention_mask=data.attention_mask)
        discourse_features = bert_output.last_hidden_state[:, 0, :]  # [CLS] token embedding for each utterance

        x = torch.cat([discourse_features, data.root_depth_embeddings], dim=1)

        x = self.gat1(x, data.edge_index, data.edge_type)
        x = nn.LeakyReLU(x)
        x = self.gat2(x, data.edge_index, data.edge_type)

        x = x.unsqueeze(0)  # Add batch dimension
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.transpose(0, 1).contiguous().view(-1, 2 * h_n.size(2))

        binary_out = self.binary_fc(h_n)
        severity_out = self.severity_fc(h_n)

        return binary_out, severity_out
