import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class GAT_MTL_Model(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_heads=4):
        super(GAT_MTL_Model, self).__init__()

        # GAT Layers for graph representation learning
        self.gat1 = GATConv(in_features, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=True)

        # Task-specific layers
        self.depression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_features)
        )

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

        # Pooling (e.g., mean pooling over nodes in each graph/conversation)
        x = global_mean_pool(x, batch)

        # Task-specific outputs
        depression_output = torch.sigmoid(self.depression_head(x))  # Binary output
        severity_output = torch.softmax(self.severity_head(x), dim=1)  # Severity classification

        return depression_output, severity_output


# Sample Data Preparation (based on embeddings from LLM-based framework)
node_features = torch.randn((10, 16))  # 10 nodes, 16-dimensional features (from LLM-based framework)
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)  # Sample edges

batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # Example batch for multiple conversations

data = Data(x=node_features, edge_index=edge_index, batch=batch)

# Initialize model, loss functions, and optimizer
model = GAT_MTL_Model(in_features=16, hidden_dim=32, out_features=4)
depression_criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for depression confirmation
severity_criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for severity classification

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# Training Loop with enhanced loss balancing
model.train()
for epoch in range(50):
    optimizer.zero_grad()

    # Forward pass
    depression_output, severity_output = model(data)

    # Define labels (example labels, replace with actual data)
    depression_label = torch.tensor([1.0, 1.0, 1.0])  # Assume all graphs have confirmed depression
    severity_label = torch.tensor([0, 2, 3])  # Severity levels: minimal, moderate, severe

    # Calculate task-specific losses
    loss_depression = depression_criterion(depression_output.squeeze(), depression_label)
    loss_severity = severity_criterion(severity_output, severity_label)

    # Combine losses with dynamic weighting based on epoch
    alpha, beta = 0.6, 0.4  # Modify weights per task if needed
    total_loss = alpha * loss_depression + beta * loss_severity

    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/50], Loss: {total_loss.item():.4f}")

# Inference example
model.eval()
with torch.no_grad():
    depression_pred, severity_pred = model(data)
    print("Depression Confirmation Prediction:", depression_pred)
    print("Severity Prediction:", severity_pred.argmax(dim=1))  # Predicted severity levels
