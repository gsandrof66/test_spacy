# https://medium.com/@whyamit101/fine-tuning-bert-for-named-entity-recognition-ner-b42bcf55b51d
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset, Dataset
import json, pandas as pd

model_name = "answerdotai/ModernBERT-base"
model_dir =  '.model/'

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)

def read_json(file: str):
    with open(file, 'r') as f:
        data = json.load(f)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(data)
    return dataset

# def test_index(sentence: str, substring: str):
#     start_index = sentence.find(substring)
#     end_index = start_index + len(substring)
#     print(f"Start: {start_index}, End: {end_index}")
#     return (start_index, end_index)
# test_index("Administer 10ml of Amoxicillin twice a day.", "twice a day")


dataset = read_json('./demo_transformers/dataset.json')
df = pd.json_normalize(dataset)
# print(dataset)
print(df)

label_list = ["DOSAGE", "MEDICATION", "FREQUENCY"]
label_map = {label: i for i, label in enumerate(label_list)}

# Tokenize and align labels
def tokenize_n_align_labels(examples):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, is_split_into_words=False)
    labels = []
    for i, label in enumerate(examples["entities"]):
        print(f"sentence: {examples['sentence'][i]}")
        word_ids = tokenized_inputs.word_ids(batch_index = i)
        print(f"tokenizer: {tokenized_inputs["input_ids"][i]}")
        print(f"word_ids: {word_ids}")
        label_ids = [-100] * len(tokenized_inputs["input_ids"][i])
        print(f"label_ids: {label_ids}")
        for entity in label:
            start, end, entity_label = entity["start"], entity["end"], entity["label"]
            print(f"start: {start}, end: {end}, entity_label: {entity_label}")
            for word_id in range(start, end):
                print(f"word_id: {word_id}")
                if word_id is not None:
                    # label_ids[word_id] = label_map[entity_label]
                    print(f"label_map: {label_map[entity_label]}")
    #     labels.append(label_ids)
    # tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_align_labels(examples):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"entities"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, is_split_into_words=False)
    labels = []
    for i, label in enumerate(examples["entities"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized_inputs["input_ids"][i])
        for entity in label:
            start, end, entity_label = entity["start"], entity["end"], entity["label"]
            for word_id in range(start, end):
                if word_id is not None and word_id < len(label_ids):
                    label_ids[word_id] = label_map[entity_label]
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
print(tokenized_dataset['labels'])

# model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=model_dir, num_labels=len(label_list))

# # Set training arguments
# training_args = TrainingArguments(
#     output_dir="./test_1",
#     # evaluation_strategy="epoch",
#     eval_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )

# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
# )

# trainer.train()

