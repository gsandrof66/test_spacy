from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd

# local folder
model_dir =  '.model/'

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", cache_dir=model_dir)
model = AutoModelForTokenClassification.from_pretrained("answerdotai/ModernBERT-base", cache_dir=model_dir)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

text = """The patient is diagnosed with Alzheimer's disease. 
She is Vitamin D deficient.The patient is 35 years old. The age of the patient is 35 years.
The patient was prescribed 500mg of Amoxicillin. Patient should take 5 mg of Prednisone daily.
"""

ner_results = ner_pipeline(text)

df = pd.DataFrame(ner_results)
print(df)