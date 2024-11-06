import os

import pandas as pd
import torch
from torch_geometric.data import Data

from mpc_to_graph import MPCGraph


def pretrain_depression_detection(model, data, depression_criterion, optimizer):
    optimizer.zero_grad()
    depression_output, _ = model(data)
    depression_label = data.y_depression  # Assuming depression labels are in data

    # Calculate depression loss
    loss = depression_criterion(depression_output.squeeze(), depression_label.float())

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    return loss.item()


def fine_tune_severity_classification(model, data, severity_criterion, optimizer):
    optimizer.zero_grad()
    depressed_data = filter_depressed_utterances(data)  # Filtered data using your framework’s output
    _, severity_output = model(depressed_data)
    severity_label = depressed_data.y_severity  # Severity labels only for depressed utterances

    # Calculate severity classification loss
    loss = severity_criterion(severity_output, severity_label)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    return loss.item()


def filter_depressed_utterances(data):
    # The model's data includes a way to filter for depressed utterances
    depressed_mask = data.depressed_mask
    filtered_data = MPCGraph.create_mpc_graph(x=data.x[depressed_mask], edge_index=data.edge_index[:, depressed_mask], device=data.device)
    filtered_data.y_severity = data.y_severity[depressed_mask]
    return filtered_data


def create_severity_data(severity_samples, device):
    data_list = []
    for sample in severity_samples:
        utterance_vector = torch.rand((16,)).to(device)
        y_severity = sample['severity_label']

        x = utterance_vector.unsqueeze(0).to(device)
        y_severity = torch.tensor(y_severity, dtype=torch.long).unsqueeze(0).to(device)

        data_list.append(Data(x=x, y_severity=y_severity))

    return data_list


def training_args():
    learning_rate = 1e-3
    decay_factor = 1e-4
    filter_data = False
    batch_size = 32
    shuffle = True
    epochs = 50
    return batch_size, decay_factor, epochs, filter_data, learning_rate, shuffle


def load_data(source, text, label_1, label_2=None):
    extracted_data = {}
    for root, _, files in os.walk(source):
        for file in files:
            data = pd.read_csv(os.path.join(root, file))
            extracted_data[file.split('.')[0]] = [data[text].tolist(), data[label_1].tolist(),
                                               data[label_2].tolist() if label_2 else []]
    return extracted_data
