# Simple **SPACY** example
This example shows a simple pipeline to create a NER application

# Available models
You need to download one of the available models:
## English Models
* en_core_web_sm: Small model optimized for CPU, suitable for basic tasks.
* en_core_web_md: Medium model with word vectors, suitable for more complex tasks.
* en_core_web_lg: Large model with extensive word vectors, suitable for high accuracy tasks.
- You can install a model with this code: python -m spacy download en_core_web_sm

## Considerations
Spacy is not compatible with Python 3.13 (April 2025)
Ref: https://github.com/explosion/spaCy/issues/13658
