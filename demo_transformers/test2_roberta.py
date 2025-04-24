from transformers import RobertaTokenizer, RobertaForTokenClassification
from transformers import pipeline

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForTokenClassification.from_pretrained("roberta-base")

# NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

text = "The patient was prescribed 100mg of ibuprofen for pain relief and 500mg of amoxicillin for the infection."
print(text)
ner_results = ner_pipeline(text)

for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
