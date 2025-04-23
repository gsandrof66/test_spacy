import spacy
from spacy.training.example import Example

# Load blank model
nlp = spacy.blank("en")

# Add NER pipeline
ner = nlp.add_pipe("ner")

# Add labels
ner.add_label("ORG")
ner.add_label("GPE")

# Training data
TRAIN_DATA = [
    ("Apple is looking at buying U.K. startup for $1 billion", {"entities": [(0, 5, "ORG"), (27, 30, "GPE")]}),
    ("San Francisco considers banning sidewalk delivery robots", {"entities": [(0, 13, "GPE")]}),
]

# Training loop
optimizer = nlp.begin_training()
for i in range(20):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], sgd=optimizer)

# Save the model
nlp.to_disk("./ner_custom_model")
