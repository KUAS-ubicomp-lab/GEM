import logging

import torch.optim as optim
from torch import nn
from torch_geometric.data import DataLoader

from GAT_MTL import GAT_MTL_Model
from MTL_utils import pretrain_depression_detection, fine_tune_severity_classification, filter_depressed_utterances, \
    training_args, create_severity_data, load_data
from mpc_to_graph import MPCGraph

logger = logging.getLogger()


def train(model, learning_rate, decay_factor, mpc_data_loader, filtered_data_loader, severity_data_loader,
          utterances,
          filter_data, epochs):
    depression_criterion = nn.CrossEntropyLoss()
    severity_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_factor)
    model.train()

    total_loss = 0
    for epoch in range(epochs):
        print("Starting Pretraining for Depression Detection...")
        total_loss += pretrain_depression_detection(model, mpc_data_loader, depression_criterion, optimizer)
        print(f"Pretrain Epoch {epoch + 1}/{epochs}, Depression Loss: {total_loss / len(mpc_data_loader):.4f}")
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

    (batch_size, decay_factor, epochs, filter_data, learning_rate, shuffle, in_channels, hidden_channels,
     lstm_hidden_size, num_classes_depression, num_classes_severity, device) = training_args()

    model = GAT_MTL_Model(in_channels=in_channels,
                          hidden_channels=hidden_channels,
                          lstm_hidden_size=lstm_hidden_size,
                          num_classes_binary=num_classes_depression,
                          num_classes_severity=num_classes_severity,
                          ).to(device=device)

    mpc_graph = MPCGraph()
    mpc_data_graph = mpc_graph.create_hierarchical_graph(conversations=utterances, device=device)
    mpc_data_loader = DataLoader(mpc_data_graph, batch_size=batch_size, shuffle=shuffle)

    print("Filtering for Depressed Utterances...")
    filtered_data = filter_depressed_utterances(data=mpc_data_graph, model=model, threshold=0.5)
    filtered_data_loader = DataLoader(filtered_data, batch_size=batch_size, shuffle=shuffle)

    print("Creating Severity data...")
    samples = list(load_data(source='severity_data', text='text', label_1='label').values())[0]
    severity_samples = list(zip(samples[0], samples[1]))

    severity_data_list = create_severity_data(severity_samples=severity_samples,
                                              device=device)
    severity_data_loader = DataLoader(severity_data_list, batch_size=batch_size, shuffle=shuffle)

    train(model=model,
          learning_rate=learning_rate,
          decay_factor=decay_factor,
          mpc_data_loader=mpc_data_graph,
          filtered_data_loader=filtered_data_loader,
          severity_data_loader=severity_data_loader,
          utterances=utterances,
          filter_data=filter_data,
          epochs=epochs)


if __name__ == '__main__':
    main()
