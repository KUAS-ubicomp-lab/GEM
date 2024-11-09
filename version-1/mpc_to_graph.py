import torch
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel


class MPCGraph(Data):
    def __init__(self, hidden_size=768):
        super(MPCGraph, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.bert_model.config.hidden_size

    def create_hierarchical_graph(self, conversations, device):
        """
        Convert MPC data into a graph structure with root-sub and sub-sub level edges.
         """
        node_features = []
        node_labels = []
        edge_index = []
        edge_type = []
        utterance_id_to_idx = {}

        for conversation in conversations:
            for idx, utterance in enumerate(conversation):
                utterance_embedding = self.encode_utterance(utterance['text'])
                node_features.append(utterance_embedding)

                depression_label = utterance.get('depression_label', 0)
                node_labels.append(depression_label)

                current_node_idx = len(node_features) - 1
                utterance_id_to_idx[utterance['utterance_id']] = current_node_idx

                parent_id = utterance.get('parent_id')
                if parent_id:
                    # Root-Sub or Sub-Sub connection
                    parent_idx = utterance_id_to_idx.get(parent_id)
                    if parent_idx is not None:
                        # Create root-sub (edge type 0) or sub-sub (edge type 1) edge
                        edge_index.append([parent_idx, current_node_idx])
                        edge_type.append(0 if idx == 0 else 1)

                # Additional Speaker and Temporal Relationships edge types (edge type 2 and 3)
                self.create_mpc_graph(conversation, current_node_idx, edge_index, edge_type, idx, utterance,
                                      utterance_id_to_idx)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device=device)
        edge_type = torch.tensor(edge_type, dtype=torch.long).to(device=device)
        node_features = torch.stack(node_features).to(device=device)
        node_labels = torch.tensor(node_labels, dtype=torch.long).to(device=device)

        return Data(x=node_features, edge_index=edge_index, edge_type=edge_type, y=node_labels)

    def create_mpc_graph(self, conversation, current_node_idx, edge_index, edge_type, idx, utterance,
                         utterance_id_to_idx):
        for other_utterance in conversation[:idx]:
            if other_utterance['speaker'] == utterance['speaker']:
                # Same-speaker relationship (edge type 2)
                other_idx = utterance_id_to_idx[other_utterance['utterance_id']]
                edge_index.append([other_idx, current_node_idx])
                edge_type.append(2)

            # Temporal proximity (edge type 3)
            if abs(utterance['timestamp'] - other_utterance['timestamp']) < 5:
                other_idx = utterance_id_to_idx[other_utterance['utterance_id']]
                edge_index.append([other_idx, current_node_idx])
                edge_type.append(3)

    def encode_utterance(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        if cls_embedding.size(1) != self.hidden_size:
            cls_embedding = cls_embedding[:, :self.hidden_size]
        return cls_embedding.squeeze(0)

    def extract_features(self, utterance):
        token_embedding = self.encode_utterance(utterance['text'])
        speaker_embedding = torch.zeros(4)
        if 'speaker' in utterance:
            speaker_id = hash(utterance['speaker']) % 4
            speaker_embedding[speaker_id] = 1

        feature_vector = torch.cat((token_embedding, speaker_embedding), dim=0)
        return feature_vector
