import json
import os
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

DATASET_PATH = os.path.join(os.path.dirname(__file__), "user_interactions.jsonl")
LABELS = ["REAL", "HOAX"]

# Load and label data
def load_labeled_data():
    data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # Add a label field if not present (default: None)
            label = entry.get("label", None)
            if label is not None:
                data.append({"text": entry["user"], "label": label})
    return data

def label_data():
    # Interactive labeling for new entries
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        entry = json.loads(line)
        if "label" not in entry:
            print(f"Text: {entry['user']}")
            label = input(f"Label ({'/'.join(LABELS)}): ").strip().upper()
            if label in LABELS:
                entry["label"] = label
        new_lines.append(json.dumps(entry, ensure_ascii=False))
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

# Training function
def train_model():
    data = load_labeled_data()
    if not data:
        print("No labeled data found.")
        return
    texts = [d["text"] for d in data]
    labels = [LABELS.index(d["label"]) for d in data]
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(texts, truncation=True, padding=True)
    import torch
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
    dataset = Dataset(encodings, labels)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(LABELS))
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()
    model.save_pretrained("./finetuned_llm")
    tokenizer.save_pretrained("./finetuned_llm")
    print("Model fine-tuned and saved to ./finetuned_llm")

if __name__ == "__main__":
    print("Choose: label (l) or train (t)")
    mode = input("Mode: ").strip().lower()
    if mode == "l":
        label_data()
    elif mode == "t":
        train_model()
    else:
        print("Unknown mode.")
