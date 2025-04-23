import spacy

# Load the med7 model
med7 = spacy.load("en_core_med7_lg")

# Sample medical text
text = "The patient was prescribed 100mg of ibuprofen for pain relief and 500mg of amoxicillin for the infection."

# Process the text
doc = med7(text)

# Print the named entities
for ent in doc.ents:
    print(ent.text, ent.label_)
