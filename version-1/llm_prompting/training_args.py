import os
import json
import argparse
import dataclasses
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments, HfArgumentParser


@dataclass
class TrainingArguments(TrainingArguments):
    out_dir_root: str = field(
        default='outputs', metadata={"help": "Path to save checkpoints and results"}
    )
    backbone: str = field(
        default='wsw', metadata={"help": "Path to pretrained model or model identifier from "
                                         "huggingface.co/models"}
    )
    prompt_len: int = field(
        default=50, metadata={"help": "Length of soft prompt tokens."}
    )
    num_of_demonstrations: int = field(
        default=5, metadata={"help": "Number of demonstrations."}
    )
    demonstration_type: Optional[str] = field(
        default='prompt_tuning',
        metadata={"help": "The type of the demonstrations to be used as in-context demonstrations."}
    )
    num_proj_layers: int = field(
        default=1, metadata={"help": "Number of layers of the cross-model projector."}
    )
    flatten_proj: bool = field(
        default=True, metadata={"help": "Whether to flatten prompt embeddings for cross-model projector."}
    )
    dataset: Optional[str] = field(
        default='rsdd', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    source_dataset: Optional[str] = field(
        default='eRisk18T2', metadata={"help": "The name of the source dataset to use (via the datasets library)."}
    )
    target_dataset: Optional[str] = field(
        default='eRisk22T2',
        metadata={"help": "The name of the target dataset to use (via the datasets library)."}
    )
    checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint."}
    )
    max_source_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    model_parallel: bool = field(
        default=False, metadata={"help": "Whether to use model parallelism."}
    )

    def __post_init__(self):
        super().__post_init__()

        self.dataset = self.dataset.lower()
        self.source_dataset = self.dataset
        self.output_dir = os.path.join(self.output_dir, self.dataset)


class RemainArgHfArgumentParser(HfArgumentParser):
    def parse_json_file(self, json_file: str, return_remaining_args=True):
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)

        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return *outputs, remain_args
        else:
            return (*outputs,)
