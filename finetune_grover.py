from dataclasses import dataclass, field
import pandas as pd
import os
import numpy as np

from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score

import transformers
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, TrainingArguments, HfArgumentParser
from torch.utils.data import Dataset


@dataclass
class Arguments(TrainingArguments):
    dataset_folder: str = field(default=None, metadata={
        "help": "Path to the dataset folder"
    })
    seed = 42
    output_dir="results/grover_ft",
    per_device_train_batch_size=16
    eval_strategy="epoch"
    learning_rate=0.000001
    num_train_epochs=4


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, texts, labels, tokenizer):

        super(SupervisedDataset, self).__init__()

        sequences = [text for text in texts]

        output = tokenizer(
            sequences,
            add_special_tokens=True,
            max_length=310,
            padding="longest",
            return_tensors="pt",
            truncation=True
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i]
        )


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": matthews_corrcoef(
            labels, predictions
        ),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def get_dataset(path):
    data = pd.read_csv(path)

def main(args: Arguments): 
    assert args.dataset_folder

    train_path = f"{args.dataset_folder}/train.csv"
    val_path = f"{args.dataset_folder}/dev.csv"
    test_path = f"{args.dataset_folder}/test.csv"

    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    tokenizer = AutoTokenizer.from_pretrained("PoetschLab/GROVER")
    model = AutoModelForSequenceClassification.from_pretrained(
        "PoetschLab/GROVER",
        num_labels=len(train_data["label"].unique())
    )

    train_dataset = SupervisedDataset(train_data["sequence"], train_data["label"], tokenizer)
    test_dataset = SupervisedDataset(test_data["sequence"], test_data["label"], tokenizer)
    val_dataset = SupervisedDataset(validation_data["sequence"], validation_data["label"], tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args
    )
    trainer.train()

    results = trainer.evaluate(eval_dataset=test_dataset)
    print(results)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)