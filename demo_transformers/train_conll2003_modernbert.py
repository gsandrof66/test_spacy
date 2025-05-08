from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForTokenClassification
import tf_keras as keras
from datasets import load_dataset, Dataset
from collections import Counter
import json, pandas as pd
# from seqeval.metrics import precision_score, recall_score, f1_score

# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     true_labels = [[id2label[l] for l in label] for label in labels]
#     true_preds = [[id2label[p] for p in pred] for pred in preds]
    
#     precision = precision_score(true_labels, true_preds)
#     recall = recall_score(true_labels, true_preds)
#     f1 = f1_score(true_labels, true_preds)
#     return {"precision": precision, "recall": recall, "f1": f1}

# Smodel_name = "answerdotai/ModernBERT-base"
model_name = "google-bert/bert-base-cased"
model_dir =  '.model/'

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)

dataset = load_dataset("tner/conll2003")
print(dataset)

# Convert dataset into BERT-compatible format
def tokenize_and_align_labels(batch, tokenizer):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(batch["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id != previous_word_id:
                label_ids.append(label[word_id] if word_id is not None else -100)
            else:
                label_ids.append(label[word_id] if word_id is not None else -100)
            previous_word_id = word_id
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

# Inspect class distribution
train_labels = [label for sample in dataset["train"]["tags"] for label in sample]
label_counts = Counter(train_labels)
# print(f"Class distribution: {label_counts}")

# Load pre-trained BERT model with a token classification head
model = AutoModelForTokenClassification.from_pretrained(
    model_name, cache_dir=model_dir,
    num_labels=len(label_counts),  # Adjust for the number of entity types in your dataset
)

# Check the model architecture
# print(model.config)

training_args = TrainingArguments(
    output_dir="./my_modernbert",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=10,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,  # Custom function to calculate F1, precision, recall
)

trainer.train()
