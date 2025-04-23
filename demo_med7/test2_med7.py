import spacy

med7 = spacy.load("en_core_med7_lg")

# create distinct colours for labels
col_dict = {}
seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
    col_dict[label] = colour

print(col_dict)

options = {'ents': med7.pipe_labels['ner'], 'colors':col_dict}

text = 'A patient was prescribed Magnesium hydroxide 400mg/5ml suspension PO of total 30ml bid for the next 5 days.'
# doc = med7(text)

# Process texts
docs = [med7(tx) for tx in text]

# Extract entities
for doc in docs:
    for ent in doc.ents:
        print(ent.text, ent.label_)

# spacy.displacy.render(doc, style='ent', jupyter=True, options=options)
# [(ent.text, ent.label_) for ent in doc.ents]