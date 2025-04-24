from transformers import RobertaTokenizer, RobertaForTokenClassification
import torch

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=9)  # Adjust num_labels based on your dataset

# Example text
text = "The patient was prescribed 100mg of ibuprofen for pain relief and 500mg of amoxicillin for the infection."

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt")

# Get model predictions
outputs = model(**inputs)
logits = outputs.logits

# Convert logits to predicted labels
predictions = torch.argmax(logits, dim=2)

# Map predictions to labels
label_map = {0: "O", 1: "B-MEDICATION", 2: "I-MEDICATION", 3: "B-DOSAGE", 4: "I-DOSAGE", 5: "B-CONDITION", 6: "I-CONDITION", 7: "B-PATIENT", 8: "I-PATIENT"}  # Example label map
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = [label_map[pred.item()] for pred in predictions[0]]

# Print tokens with their predicted labels
print(text)
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")
