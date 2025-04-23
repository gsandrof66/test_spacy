import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

# Load the existing model
nlp = spacy.load("en_core_med7_lg")

# Add the new entity labels to the model
ner = nlp.get_pipe("ner")
for label in ["NEW_ENTITY_1", "NEW_ENTITY_2"]:  # Replace with your custom labels
    ner.add_label(label)

# Prepare your training data
TRAIN_DATA = [
    ("Your annotated text here", {"entities": [(start, end, label)]}),
    # Add more examples
]

# Convert training data to spaCy format
examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in TRAIN_DATA]

# Training loop
optimizer = nlp.resume_training()
for i in range(10):  # Number of iterations
    losses = {}
    batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, drop=0.5, losses=losses)
    print(f"Losses at iteration {i}: {losses}")

# Save the updated model
nlp.to_disk("./demo_med7/custom_med7_model")
