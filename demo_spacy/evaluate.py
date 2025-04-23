import spacy

# Load the saved model
nlp = spacy.load("./ner_custom_model")

# Test data
test_text = "Google is opening a new office in New York"

# Process the text
doc = nlp(test_text)

# Print entities
for ent in doc.ents:
    print(ent.text, ent.label_)
