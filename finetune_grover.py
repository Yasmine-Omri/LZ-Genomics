from dataclasses import dataclass, field
import time
import pandas as pd
import os
import numpy as np

from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score

import transformers
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, AutoModelForSequenceClassification, TrainingArguments, HfArgumentParser
from torch.utils.data import Dataset
from memory_profiler import profile


@dataclass
class Arguments(TrainingArguments):
    dataset_folder: str = field(default=None, metadata={
        "help": "Path to the dataset folder"
    })
    seed: int = field(default=42)
    output_dir: str = field(default="results/grover_ft")
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=8)  # for 32 total batch size with 4 GPUs
    eval_strategy: str = field(default="steps")
    learning_rate: float = field(default=1e-4) # from grover paper
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=50)
    num_train_epochs: int = field(default=8)
    metric_for_best_model: str = field(default="matthews_correlation")
    load_best_model_at_end: bool = field(default=True)
    save_only_model: bool = field(default=True)



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, texts, labels, tokenizer):

        super(SupervisedDataset, self).__init__()

        sequences = [text for text in texts]

        output = tokenizer(
            sequences,
            add_special_tokens=True,
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

@profile
def main(args: Arguments): 
    assert args.dataset_folder

    read_data_in_time = time.perf_counter()

    print("-----READING DATA")
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

    train_start_time = time.perf_counter()
    print("-----TRAINING")
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args
    )
    trainer.train()

    train_end_time = time.perf_counter()
    print("-----EVALUATION")
    results = trainer.evaluate(eval_dataset=test_dataset)
    inference_end_time = time.perf_counter()
    print(results)

    print(f"TEST ACCURACY: {results['eval_accuracy']}")
    print(f"TEST F1: {results['eval_f1']}")
    print(f"TEST PRECISION: {results['eval_precision']}")
    print(f"TEST RECALL: {results['eval_recall']}")
    print(f"TEST MCC: {results['eval_matthews_correlation']}")

    print("-----TIME PROFILING+")
    # Read train + val data time: from initial read start to training start
    print(f"Read train + val data time: {(train_start_time - read_data_in_time): .5f}")

    # Training symbols & seq length estimate
    try:
        nb_train_seqs = len(train_data)
        train_seq_len = len(train_data.iloc[0]["sequence"])
        nb_train_symbols = nb_train_seqs * train_seq_len
    except Exception:
        nb_train_seqs = len(train_data); train_seq_len = 0; nb_train_symbols = 0

    train_duration = train_end_time - train_start_time

    print(f"Number of training symbols: {nb_train_symbols}")
    print(f"Length of one training sequence: {train_seq_len}")
    print(f"Total training time: {train_duration:.3f} seconds")

    # Test timing: emulate Train.py's 'Read test data time'
    try:
        nb_test_seqs = len(test_data)
        test_seq_len = len(test_data.iloc[0]["sequence"])
    except Exception:
        nb_test_seqs = len(test_data); test_seq_len = 0

    read_test_data_start_time = time.perf_counter()
    _ = pd.read_csv(test_path)
    read_test_data_end_time = time.perf_counter()  # mimic explicit read step
    

    print(f"Number of test sequences: {nb_test_seqs}")
    print(f"Length of test sequence: {test_seq_len}")
    print(f"Read test data time: {(read_test_data_end_time - read_test_data_start_time): .5f}")
    inference_duration = (inference_end_time - train_end_time)
    print(f"Total inference time: {inference_duration:.3f} seconds")
    den = (nb_test_seqs * test_seq_len) if (nb_test_seqs and test_seq_len) else 0
    print(f"Inference time/symbol: {inference_duration/den if den else 0.0} seconds")
    # ---------- END TIME PROFILING+ ----------


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)