import random

import torch
from torch.utils.data import DataLoader

from training_args import TrainingArguments
from ..GAT_MTL import GAT_MTL_Model
from ..MTL_utils import load_data, training_args
from ..modeling_llms.modeling_llama import LlamaForSequenceClassification
from ..modeling_llms.modeling_wsw import WSWEmbeddings

models = {
    'GAT_MTL': GAT_MTL_Model,
    'WSW': WSWEmbeddings,
    'LLaMA': LlamaForSequenceClassification
}

input_mpc_depressed_data_list = (
        list(load_data(source='mpc_data', text='tweet', label_1='conversation_id', label_2='user_id').values()))
datasets = {'RSDD': TrainingArguments.dataset, 'MPCDataset': input_mpc_depressed_data_list}

settings = ['full-shot', 'few-shot', 'zero-shot']
device = training_args()


def main_evaluation():
    results = {}
    for dataset_name, dataset in datasets.items():
        for model_name, model_class in models.items():
            for setting in settings:
                model = model_class().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

                if setting == 'full-shot':
                    train_loader, val_loader, test_loader = create_data_loaders(dataset,
                                                                                setting='full')
                elif setting == 'few-shot':
                    train_loader, val_loader, test_loader = create_data_loaders(dataset,
                                                                                setting='few',
                                                                                few_shot_percentage=0.05)
                elif setting == 'zero-shot':
                    test_loader = DataLoader(dataset['test'],
                                             batch_size=4,
                                             shuffle=False)

                if setting != 'zero-shot':
                    train_model(model, train_loader, optimizer)

                metrics = evaluate_model(model, test_loader)
                results[(dataset_name, model_name, setting)] = metrics
                print(f"{dataset_name} | {model_name} | {setting} - Metrics: {metrics}")

    save_results(results)


def create_data_loaders(dataset, setting='full', few_shot_percentage=1.0):
    if setting == 'few':
        dataset = random.sample(dataset, int(len(dataset) * few_shot_percentage))
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, optimizer, epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            outputs = model(data.to(device))
            loss = criterion(outputs, data.y.to(device))
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_loader):
    model.eval()
    binary_metrics, severity_metrics = [], []
    with torch.no_grad():
        for data in test_loader:
            outputs = model(data.to(device))
    return {'binary_metrics': binary_metrics, 'severity_metrics': severity_metrics}


def save_results(results):
    pass


main_evaluation()
