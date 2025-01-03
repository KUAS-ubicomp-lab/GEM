import os
from typing import Optional

import torch
import torch.nn as nn
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, SoftVerbalizer
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

from data_processor import data_processor_list
from training_args import TrainingArguments
from ..modeling_llms.modeling_llama import LlamaForSequenceClassification
from ..modeling_llms.modeling_wsw import WSWEmbeddings


class PromptKernel(Trainer):

    def __init__(self, prompt_emb=None, **kwargs):
        args = kwargs['args']
        args.demonstration_type = "prompt_tuning"
        self.prompt_emb = prompt_emb
        self.demonstration_n = args.demonstration_sample
        self.free_text_explanations = args.free_text_explanations
        self.latent_dropout = args.latent_dropout
        self.out_dir_root = args.output_dir
        self.pt_demonstration_input_layer = args.pt_demonstration_input_layer
        self.pt_demonstration_output_layer = args.pt_demonstration_output_layer
        self.pt_activation = args.backbone.activation

        processor = data_processor_list[args.dataset]()

        # Model
        template_text = '{"soft": None, "duplicate": ' + str(args.prompt_len) + ', "same": True} {"mask"} {' \
                                                                                '"placeholder": "text_a"} {' \
                                                                                '"placeholder": "text_b"}'

        model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, model_type = self.get_model(
            args.backbone, processor, args)

        self.set_active_state_dict(model)  # Only save soft prompts

        # Initialize transformers.trainer
        kwargs['model'] = model
        kwargs['tokenizer'] = tokenizer
        kwargs['train_dataset'] = processor.train_dataset
        kwargs['eval_dataset'] = processor.eval_dataset
        super().__init__(**kwargs)

        self.config = model_config
        self.plm = plm
        self.template_text = template_text
        self.template = template
        self.verbalizer = verbalizer
        self.tokenizer_wrapper_class = tokenizer_wrapper_class
        self.prompt_emb = template.soft_embeds

        # Soft prompt transfer
        self.source_model_type = model_type
        self.target_model_type = None

        print('Trainable parameters:')
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n, p.shape)

        print(f'Template: {self.template}')
        print(f'Verbalizer: {self.verbalizer}')
        print(f'Raw input example: {self.train_dataset[0]}')
        print(f'Wrapped input example: {self.template.wrap_one_example(self.train_dataset[0])}')

    def get_model(self, model_name, processor, args):
        model_type = []

        if 'bert-' in model_name:
            model_type = 'bert'

        elif 'wsw-' in model_name:
            model_type = 'wsw'

        elif 'llama-' in model_name:
            model_type = 'llama'

        # Load openprompt models
        if (not hasattr(self, 'args')) or args.backbone != model_name:
            plm, tokenizer, model_config, tokenizer_wrapper_class = load_plm(model_type, model_name)
        else:
            plm = self.plm
            tokenizer = self.tokenizer
            model_config = self.config
            tokenizer_wrapper_class = self.tokenizer_wrapper_class
            model_type = self.source_model_type

        # Load soft template
        if hasattr(self, 'template') and self.template is not None:
            template = self.template
        else:
            template_text = '{"soft": None, "duplicate": ' + str(
                args.prompt_len) + ', "same": True} {"mask"} {"placeholder": "text_a"} {"placeholder": "text_b"}'
            template = SoftTemplate(model=plm, tokenizer=tokenizer, text=template_text,
                                    soft_embeds=self.get_prompt_emb(args, model_config), num_tokens=args.prompt_len)

        # Load soft verbalizer. This is the first time of using a soft verbalizer in prompt task transferring to our
        # knowledge.
        if hasattr(self, 'verbalizer') and self.verbalizer is not None:
            verbalizer = self.verbalizer
        else:
            verbalizer = SoftVerbalizer(tokenizer, model=plm, classes=processor.labels,
                                        label_words=processor.label_words)

        # Set In-Context Demonstrations (ICD). This is the first time of using ICD in out-of-domain task transfer to
        # our knowledge.
        if hasattr(self, 'demonstration_type') and args.demonstration_type is not None:
            self.set_in_context_demonstrations(demonstration_type=self.demonstration_type,
                                               demonstration_sample=self.get_prompt_emb(args, model_config))

        if hasattr(self, 'model') and self.model is not None:
            model = self.model
        else:
            model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=True)

            if hasattr(args, "model_parallel") and args.model_parallel:
                print('parallelize model!')
                model.parallelize()

        _keys_to_ignore_on_save = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                _keys_to_ignore_on_save.append(n)

        model._keys_to_ignore_on_save = _keys_to_ignore_on_save

        return model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, model_type

    @torch.no_grad()
    def get_prompt_emb(self, args, config):
        prompt_emb = []

        if 'bert-' in args.backbone:
            prompt_emb = args.backbone.bert.embeddings.prompt_embeddings

        if 'wsw-' in args.backbone:
            prompt_emb = WSWEmbeddings(config=config)

        if 'llama-' in args.backbone:
            prompt_emb = LlamaForSequenceClassification(config=config)

        return prompt_emb.prompt_embeddings.weight

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataloader = PromptDataLoader(
            dataset=self.train_dataset,
            template=self.template,
            verbalizer=self.verbalizer,
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.tokenizer_wrapper_class,
            batch_size=self._train_batch_size,
            max_seq_length=self.args.max_source_length,
            decoder_max_length=1,
            shuffle=True)

        return train_dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        validation_dataloader = PromptDataLoader(
            dataset=eval_dataset,
            template=self.template,
            verbalizer=self.verbalizer,
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.tokenizer_wrapper_class,
            batch_size=self.args.per_device_eval_batch_size,
            max_seq_length=self.args.max_source_length,
            decoder_max_length=1,
            device=torch.device("cuda:1"),
            shuffle=False)

        return validation_dataloader

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = nn.CrossEntropyLoss()(outputs, inputs['label'])

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def set_active_state_dict(module: nn.Module, includes=['prompt_model.template.soft_embeds']):
        def _caller(_org_func, includes, *args, **kwargs):
            state_dict = _org_func(*args, **kwargs)
            keys = list(state_dict.keys())
            for n in keys:
                if n not in includes:
                    state_dict.pop(n)
            return state_dict

        includes = includes
        if hasattr(module.state_dict, "__wrapped__"):
            raise RuntimeWarning(
                "The forward function might have been wrapped by a decorator, is it intended? Do you freeze the "
                "parameters twice?")

    def set_in_context_demonstrations(self, demonstration_type, demonstration_sample):
        self.demonstration_n = demonstration_sample
        self.latent_dropout = nn.Dropout(0.1)
        self.config.demonstration_type = demonstration_type
        self.free_text_explanations = None

        if TrainingArguments.demonstration_type == "prompt_tuning":
            self.demonstration_n = demonstration_sample
        else:
            raise NotImplementedError

        self.prompt_emb = nn.Embedding(demonstration_sample, nn.Embedding.embedding_dim)
        nn.LayerNorm(nn.Embedding.embedding_dim, eps=self.config.layer_norm_eps)
        self.prompt_emb.weight = self.prompt_emb
        self.pt_demonstration_input_layer = nn.Linear(nn.Embedding.embedding_dim, nn.Embedding.embedding_dim * 4)
        self.pt_activation = nn.GELU
        self.pt_demonstration_output_layer = nn.Linear(nn.Embedding.embedding_dim * 4, nn.Embedding.embedding_dim)
        nn.LayerNorm(nn.Embedding.embedding_dim, eps=self.config.layer_norm_eps)

    def train_prompt(self, model=None, task=None, **kwargs):
        device = torch.device('cuda:1')

        with torch.device('cuda:1'):
            self.args.output_dir = os.path.join(self.out_dir_root, 'prompt_emb')
            os.makedirs(self.args.output_dir, exist_ok=True)

            if model is None:
                model = self.model
            else:
                if isinstance(model, str):
                    if model != self.args.backbone:
                        processor = data_processor_list[task]
                        model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, \
                            model_type = self.get_model(model, processor, self.args.backbone)
                    else:
                        model = self.model
                elif isinstance(model, torch.nn.Module):
                    pass
            self._move_model_to_device(model=model, device=device)

            if task != self.args.dataset:
                processor = data_processor_list[self.args.dataset]()
                self.train_dataset = processor.train_dataset
            return super().train(**kwargs)

    def eval_prompt(self, model=None, eval_dataset=None, prompt_emb=None):
        with torch.device('cuda:1'):
            self.args.output_dir = os.path.join(self.out_dir_root, 'prompt_emb')
            os.makedirs(self.args.output_dir, exist_ok=True)

            if model is not None and isinstance(model, str) and model != self.args.backbone:
                model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, \
                    model_type = self.get_model(model, self.args.processor, self.args.backbone)
                self.model = model

            if prompt_emb is not None:
                if isinstance(prompt_emb, str):
                    prompt_emb = torch.load(prompt_emb, map_location='cpu')
                elif isinstance(prompt_emb, torch.Tensor):
                    pass
                else:
                    raise NotImplementedError
                self.args.backbone.roberta.embeddings = nn.Parameter(prompt_emb.detach())

            if eval_dataset is None or eval_dataset == self.args.dataset:
                eval_dataset = self.eval_dataset
            elif eval_dataset != self.args.dataset:  # Use a dataset different from the source task
                processor = data_processor_list[eval_dataset]()
                eval_dataset = processor.eval_dataset
            else:
                raise NotImplementedError

            metrics = self.evaluate(eval_dataset=eval_dataset)
            self.save_metrics("eval_prompt", metrics)
            return metrics
