from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import pandas as pd

# local folder
model_dir = '.model/'

model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-cased", cache_dir=model_dir)
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-cased", cache_dir=model_dir)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

text = """The patient is diagnosed with Alzheimer's disease. 
She is Vitamin D deficient.The patient is 35 years old. The age of the patient is 35 years.
The patient was prescribed 500mg of Amoxicillin. Patient should take 5 mg of Prednisone daily.
"""

# Perform NER
ner_results = ner_pipeline(text)

# for entity in ner_results:
#     print(f"Entity: {entity['word']}, Label: {entity['entity']}")

df = pd.DataFrame(ner_results)
print(df)
