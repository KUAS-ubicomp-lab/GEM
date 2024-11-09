import logging

import torch
import torch.optim as optim
from torch import nn
from torch_geometric.data import DataLoader

from GAT_MTL import GAT_MTL_Model
from MTL_utils import pretrain_depression_detection, fine_tune_severity_classification, filter_depressed_utterances, \
    training_args, create_severity_data, load_data
from mpc_to_graph import MPCGraph

logger = logging.getLogger()


def train(device, learning_rate, decay_factor, mpc_data_loader, filtered_data_loader, severity_data_loader, utterances,
          filter_data, epochs):
    # Initialize model, optimizer, and criteria
    in_features = [MPCGraph.extract_features(utterance=utterance) for utterance in utterances]
    hidden_dim, lstm_hidden_dim, out_features = 32, 64, 4
    model = GAT_MTL_Model(in_features, hidden_dim, lstm_hidden_dim, out_features)
    model.to(device)
    depression_criterion = nn.BCELoss()
    severity_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate=learning_rate, weight_decay=decay_factor)
    model.train()

    total_loss = 0
    for epoch in range(epochs):
        for data in mpc_data_loader:
            print("Starting Pretraining for Depression Detection...")
            total_loss += pretrain_depression_detection(model, data, depression_criterion, optimizer)
            print(f"Pretrain Epoch {epoch + 1}/{epochs}, Depression Loss: {total_loss / len(data):.4f}")
        print("Pretraining Complete.")

        print("Fine-tuning for Severity Classification...")
        data_loader = filtered_data_loader if filter_data else severity_data_loader
        for data in data_loader:
            total_loss += fine_tune_severity_classification(model, data, severity_criterion, optimizer)
            print(f"Fine-tune Epoch {epoch + 1}/{epochs}, Severity Loss: {total_loss / len(data):.4f}")
        print("Fine-tuning Complete.")

    avg_loss = total_loss / len(utterances)
    logger.info("Average loss =%f", avg_loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_mpc_depressed_data_list, input_mpc_non_depressed_data_list = (
        list(load_data(source='mpc_data', text='tweet', label_1='conversation_id', label_2='user_id').values()))
    # utterances = input_mpc_depressed_data_list, input_mpc_non_depressed_data_list
    utterances = [
        [
            {'utterance_id': 'u1', 'text': 'Feeling down today.', 'parent_id': None, 'speaker': 'A', 'timestamp': 1,
             'depression_label': 1},
            {'utterance_id': 'u2', 'text': 'I understand. Do you want to talk about it?', 'parent_id': 'u1',
             'speaker': 'B', 'timestamp': 2, 'depression_label': 0},
            {'utterance_id': 'u3', 'text': 'Yes, it just feels overwhelming.', 'parent_id': 'u1', 'speaker': 'A',
             'timestamp': 3, 'depression_label': 1},
        ]
    ]
    severity_data_samples = list(load_data(source='severity_data', text='text', label_1='label').values())

    batch_size, decay_factor, epochs, filter_data, learning_rate, shuffle = training_args()

    mpc_graph = MPCGraph()
    mpc_data_graph = mpc_graph.create_hierarchical_graph(conversations=utterances, device=device)
    mpc_data_loader = DataLoader(mpc_data_graph, batch_size=batch_size, shuffle=shuffle)

    print("Filtering for Depressed Utterances...")
    filtered_data = [data_filter for data_filter in mpc_data_graph if filter_depressed_utterances(data_filter)]
    filtered_data_loader = DataLoader(filtered_data, batch_size=batch_size, shuffle=shuffle)

    print("Creating Severity data...")
    severity_data_list = create_severity_data(severity_samples=severity_data_samples, device=device)
    severity_data_loader = DataLoader(severity_data_list, batch_size=batch_size, shuffle=shuffle)

    train(device=device,
          learning_rate=learning_rate,
          decay_factor=decay_factor,
          mpc_data_loader=mpc_data_loader,
          filtered_data_loader=filtered_data_loader,
          severity_data_loader=severity_data_loader,
          utterances=utterances,
          filter_data=filter_data,
          epochs=epochs)


if __name__ == '__main__':
    main()
