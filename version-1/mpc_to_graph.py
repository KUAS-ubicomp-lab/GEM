import torch
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel


class MPCGraph(Data):
    def __init__(self, hidden_size=768):
        super(MPCGraph, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def create_hierarchical_graph(self, conversations, device):
        """
        Convert MPC data into a graph structure with root-sub and sub-sub level edges.
         """
        node_features = []
        edge_index = []
        edge_type = []
        utterance_id_to_idx = {}

        for conversation in conversations:
            for idx, utterance in enumerate(conversation):
                utterance_embedding = self.encode_utterance(utterance['text'])
                node_features.append(utterance_embedding)
                utterance_id_to_idx[utterance['utterance_id']] = len(node_features) - 1

                # Create edges based on root-sub and sub-sub relationships
                parent_id = utterance.get('parent_id')
                if parent_id:
                    # Root-Sub or Sub-Sub connection
                    parent_idx = utterance_id_to_idx.get(parent_id)
                    if parent_idx is not None:
                        edge_index.append([parent_idx, len(node_features) - 1])  # Directed edge
                        edge_type.append(0 if idx == 0 else 1)  # 0 for root-sub, 1 for sub-sub

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device=device)
        edge_type = torch.tensor(edge_type, dtype=torch.long).to(device=device)
        node_features = torch.stack(node_features).to(device=device)

        return Data(x=node_features, edge_index=edge_index, edge_type=edge_type)

    def create_mpc_graph(self, conversation_data, device):
        num_utterances = len(conversation_data)
        node_features, edge_index, y_depression = [], [], []

        for i, utterance_data in enumerate(conversation_data):
            utterance_vector = self.embed_utterance(utterance_data['utterance']).to(device=device)
            node_features.append(utterance_vector)
            y_depression.append(utterance_data['depression_label'])

            # Temporal edges
            if i < num_utterances - 1:
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])

            # Speaker relationship edges
            for j in range(i + 1, num_utterances):
                if conversation_data[j]['speaker_id'] == utterance_data['speaker_id']:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        x = torch.stack(node_features).to(device=device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device=device)
        y_depression = torch.tensor(y_depression, dtype=torch.float).to(device=device)

        return Data(x=x, edge_index=edge_index, y_depression=y_depression)

    def embed_utterance(self, utterance):
        return torch.rand((16,))

    def encode_utterance(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        if cls_embedding.size(1) != self.hidden_size:
            cls_embedding = cls_embedding[:, :self.hidden_size]
        return cls_embedding.squeeze(0)
