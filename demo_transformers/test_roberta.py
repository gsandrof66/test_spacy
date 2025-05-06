from transformers import RobertaTokenizer, RobertaForTokenClassification, pipeline
import pandas as pd

model_dir =  '.model/'

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir=model_dir)
model = RobertaForTokenClassification.from_pretrained("roberta-base", cache_dir=model_dir)# Adjust num_labels based on your dataset

# NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)# , aggregation_strategy="simple" ,device=0 if using gpu

text="""
The patient is diagnosed with Alzheimer's disease. 
She is Vitamin D deficient.
The patient is 35 years old. The age of the patient is 35 years.
The patient was prescribed 500mg of Amoxicillin.
"""

ner_results = ner_pipeline(text)

# for entity in ner_results:
#     print(f"Entity: {entity['word']}, Label: {entity['entity']}")

result1_df = pd.DataFrame(ner_results)
print(result1_df)
