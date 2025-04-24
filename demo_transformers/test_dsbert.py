from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# local folder
model_dir =  '/mnt/homes/gatzos01/my_models/'

models = ["dslim/bert-base-NER", "dslim/bert-large-NER", "dslim/distilbert-NER"]
option_selected = 1
print("selecteed", models[option_selected])
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(models[option_selected], cache_dir=model_dir)
model = AutoModelForTokenClassification.from_pretrained(models[option_selected], cache_dir=model_dir)

# Create a NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Example text
text = "The patient was prescribed 500mg of Amoxicillin."

# Perform NER
ner_results = ner_pipeline(text)
print(ner_results)

for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")