# https://medium.com/@alvarani/my-foray-into-generative-ai-applying-name-entity-recognition-to-unstructured-biomedical-data-abbb82d250ff
# https://huggingface.co/d4data/biomedical-ner-all
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd

# local folder
model_dir =  '.model/'

model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all", cache_dir=model_dir)
tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all", cache_dir=model_dir)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu

text = """The patient is diagnosed with Alzheimer's disease. 
She is Vitamin D deficient.The patient is 35 years old. The age of the patient is 35 years.
The patient was prescribed 500mg of Amoxicillin. Patient should take 5 mg of Prednisone daily.
"""

ner_results   = ner_pipeline(text)

result1_df = pd.DataFrame(ner_results)
print(result1_df)