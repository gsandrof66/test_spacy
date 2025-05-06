from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# local folder
model_dir =  '.model/'

model = AutoModelForTokenClassification.from_pretrained("google-bert/bert-base-cased", cache_dir=model_dir)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased", cache_dir=model_dir)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

text="""
The patient is diagnosed with Alzheimer's disease. 
She is Vitamin D deficient.
The patient is 35 years old. The age of the patient is 35 years.
"""

# Perform NER
ner_results = ner_pipeline(text)
print(ner_results)

for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")