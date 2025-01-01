import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from openai_utils import length_of_prompt, dispatch_openai_api_requests
from prompt import PromptKernel
from training_args import TrainingArguments
from ..GAT_MTL import GAT_MTL_Model
from ..MTL_utils import load_data, training_args
from ..modeling_llms.modeling_llama import LlamaForSequenceClassification
from ..modeling_llms.modeling_wsw import WSWEmbeddings

_MAX_PROMPT_TOKENS = 25

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
                train_loader, val_loader, test_loader = [], [], []
                llm_eval = False

                if setting == 'full-shot':
                    train_loader, val_loader, test_loader = create_data_loaders(dataset,
                                                                                setting='full')
                elif setting == 'few-shot':
                    llm_eval = True
                    train_loader, val_loader, test_loader = create_data_loaders(dataset,
                                                                                setting='few',
                                                                                few_shot_percentage=0.05)
                elif setting == 'zero-shot':
                    llm_eval = True
                    test_loader = DataLoader(dataset['test'],
                                             batch_size=8,
                                             shuffle=False)

                if setting != 'zero-shot':
                    train_model(model, train_loader, optimizer)

                metrics = evaluate_llm(model, val_loader, setting) if llm_eval else evaluate_model(model, test_loader)
                results[(dataset_name, model_name, setting)] = metrics
                print(f"{dataset_name} | {model_name} | {setting} - Metrics: {metrics}")

    save_results(results)


def in_context_prediction(prompt_example, shots, engine, length_test_only=False):
    showcase_examples = [
        "{}\nQ: {}\nA: {}\n".format(s["context"], s["utterance"], s["label"]) for s in shots
    ]
    input_example = "{}\nQ: {}\nA:".format(prompt_example["context"], prompt_example["utterance"])

    demonstrations = PromptKernel.set_in_context_demonstrations("prompt_tuning",
                                                                input_example)

    prompt = "\n".join(showcase_examples + [input_example] + demonstrations)

    if length_test_only:
        prompt_length = length_of_prompt(prompt, _MAX_PROMPT_TOKENS)
        print("-----------------------------------------")
        print(prompt_length)
        print(prompt)
        return prompt_length

    response = dispatch_openai_api_requests(api_model_name=engine,
                                            prompt_list=prompt,
                                            shots=shots,
                                            max_tokens=_MAX_PROMPT_TOKENS,
                                            temperature=0.01,
                                            api_batch=len(prompt))

    prediction = response["choices"][0]
    prediction["prompt"] = prompt
    prediction["text"] = prediction["text"][len(prompt):]
    return prediction


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


def evaluate_llm(llm, val_loader, setting):
    llm.eval()
    predictions = []
    for x in tqdm(val_loader, total=len(val_loader), desc="Evaluating"):
        predictions.append(in_context_prediction(prompt_example=x,
                                                 shots=setting,
                                                 engine=llm
                                                 )
                           )
    return predictions


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
