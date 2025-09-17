import json
import os
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "Codette", "gotchu", "Codette_MemoryCocoons_FineTune.jsonl")
MODEL_NAME = "distilgpt2"

# Load data in prompt/completion format
def load_codette_data():
    data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompt = entry["prompt"]
            completion = entry["completion"]
            data.append({"prompt": prompt, "completion": completion})
    return data

def prepare_dataset(tokenizer, data):
    inputs = [d["prompt"] for d in data]
    targets = [d["completion"] for d in data]
    encodings = tokenizer(inputs, truncation=True, padding=True)
    target_encodings = tokenizer(targets, truncation=True, padding=True)
    import torch
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, targets):
            self.encodings = encodings
            self.targets = targets
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.targets["input_ids"][idx])
            return item
        def __len__(self):
            return len(self.encodings["input_ids"])
    return Dataset(encodings, target_encodings)

def train_codette_model():
    data = load_codette_data()
    if not data:
        print("No training data found.")
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = prepare_dataset(tokenizer, data)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    training_args = TrainingArguments(
        output_dir="./codette_cocoon_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_dir="./logs",
        logging_steps=5,
        save_steps=20,
        evaluation_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()
    model.save_pretrained("./codette_cocoon_model")
    tokenizer.save_pretrained("./codette_cocoon_model")
    print("Codette cocoon model trained and saved to ./codette_cocoon_model")

if __name__ == "__main__":
    train_codette_model()
