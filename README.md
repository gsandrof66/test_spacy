# **Project**
This Python project contains simple pipelines to create a NER application

# **spacy**
You need to download one of the available models:
## English Models
* en_core_web_sm: Small model optimized for CPU, suitable for basic tasks.
* en_core_web_md: Medium model with word vectors, suitable for more complex tasks.
* en_core_web_lg: Large model with extensive word vectors, suitable for high accuracy tasks.
- You can install a model with this code: python -m spacy download en_core_web_sm
- If you want to install med7 (Last updated Nov 19, 2022): pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/refs%2Fpr%2F3/en_core_med7_lg-3.4.2.1-py3-none-any.whl
. You will have more info from here: https://github.com/kormilitzin/med7
- Article about med7: https://kormilitzin.medium.com/med7-clinical-information-extraction-system-in-python-and-spacy-5e6f68ab1c68

## Considerations
- Spacy is not compatible with Python 3.13 (May 2025). Ref: https://github.com/explosion/spaCy/issues/13658
- Med 7 requires numpy 1.26.4 and spacy 3.4.4 and it is not compatible with Python 3.12

# Transformers
These models can work with sentiment analysis, named entity recognition, question answering, and text classification. There are some base models available for future evaluation
- Bert
- DistilBert
- RoBerta

There are some customised models for medical purposes
- Biomed

# Resources
Base models:
- https://huggingface.co/google-bert
- https://huggingface.co/FacebookAI

Fine tuned models:
- https://huggingface.co/Simonlee711/Clinical_ModernBERT
