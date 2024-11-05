import torch
from torch_geometric.data import Data


class MPCGraph(Data):
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
