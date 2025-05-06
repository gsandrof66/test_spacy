# https://medium.com/@whyamit101/fine-tuning-bert-for-named-entity-recognition-ner-b42bcf55b51d
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForTokenClassification

from datasets import load_dataset, Dataset
import json

model_name = "answerdotai/ModernBERT-base"
model_dir =  '.model/'

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)

def read_json(file: str):
    with open(file, 'r') as f:
        data = json.load(f)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(data)
    return dataset

dataset = read_json('./demo_transformers/dataset.json')
print(dataset)

# Tokenize dataset

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding=True)
    labels = []
    for i, label in enumerate(examples["entities"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)
        for entity in label:
            start, end, entity_label = entity["start"], entity["end"], entity["label"]
            for idx, word_id in enumerate(word_ids):
                if word_id is not None and start <= word_id < end:
                    label_ids[idx] = entity_label
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
print(tokenized_datasets)

label_list = ["DOSAGE", "MEDICATION", "FREQUENCY"]
print(label_list)

model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=model_dir, num_labels=len(label_list))

# Set training arguments
training_args = TrainingArguments(
    output_dir="./test_1",
    # evaluation_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

