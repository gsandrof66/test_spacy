# Simple **SPACY** example
This example shows a simple pipeline to create a NER application

# Available models
You need to download one of the available models:
## English Models
* en_core_web_sm: Small model optimized for CPU, suitable for basic tasks.
* en_core_web_md: Medium model with word vectors, suitable for more complex tasks.
* en_core_web_lg: Large model with extensive word vectors, suitable for high accuracy tasks.
- You can install a model with this code: python -m spacy download en_core_web_sm
- If you want to install med7: pip install "en-core-med7-lg @ https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl". You will have more info from here: https://github.com/kormilitzin/med7



## Considerations
Spacy is not compatible with Python 3.13 (April 2025)
Ref: https://github.com/explosion/spaCy/issues/13658
