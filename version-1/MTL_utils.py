import os

import pandas as pd
import torch
from torch_geometric.data import Data


def pretrain_depression_detection(model, data, depression_criterion, optimizer):
    optimizer.zero_grad()
    depression_output, _ = model.forward(data)
    depression_label = data.y  # Assuming depression labels are in data

    # Calculate depression loss
    loss = depression_criterion(depression_output, depression_label)

    # Backward pass and optimize
    loss.backward(retain_graph=True)
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


def filter_depressed_utterances(data, model, threshold):
    # The model's data includes a way to filter for depressed utterances
    out_depression, _ = model.forward(data)
    depressed_indices = (torch.softmax(out_depression, dim=1)[:, 1] > threshold).nonzero(as_tuple=True)[0]
    filtered_data = data.subgraph(depressed_indices)
    return filtered_data


def create_severity_data(severity_samples, device):
    data_list = []
    for sample in severity_samples:
        utterance_vector = torch.rand((16,)).to(device)
        y_severity = severity_mapper(sample[1])

        x = utterance_vector.unsqueeze(0).to(device)
        y_severity = torch.tensor(y_severity, dtype=torch.long).unsqueeze(0).to(device)

        data_list.append(Data(x=x, y_severity=y_severity))

    return data_list


def severity_mapper(sample):
    label_mapping = {
        "minimum": 0,
        "mild": 1,
        "moderate": 2,
        "severe": 3
    }
    return label_mapping[sample]


def training_args():
    learning_rate = 1e-3
    decay_factor = 1e-4
    filter_data = False
    batch_size = 32
    shuffle = True
    epochs = 50
    in_channels = 768
    hidden_channels = 128
    lstm_hidden_size = 64
    num_classes_depression = 2
    num_classes_severity = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (batch_size, decay_factor, epochs, filter_data, learning_rate, shuffle, in_channels, hidden_channels,
            lstm_hidden_size, num_classes_depression, num_classes_severity, device)


def load_data(source, text, label_1, label_2=None):
    extracted_data = {}
    for root, _, files in os.walk(source):
        for file in files:
            data = pd.read_csv(os.path.join(root, file))
            extracted_data[file.split('.')[0]] = [data[text].tolist(), data[label_1].tolist(),
                                                  data[label_2].tolist() if label_2 else []]
    return extracted_data
