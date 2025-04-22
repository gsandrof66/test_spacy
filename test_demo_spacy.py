import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

# Example text data
texts = ["Apple is looking at buying U.K. startup for $1 billion", "San Francisco considers banning sidewalk delivery robots"]
print(texts)

# Process texts
docs = [nlp(text) for text in texts]

# Extract entities
for doc in docs:
    for ent in doc.ents:
        print(ent.text, ent.label_)
