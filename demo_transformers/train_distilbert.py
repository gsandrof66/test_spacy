import pandas as pd
import json
import requests

train = pd.read_csv("https://raw.githubusercontent.com/Syahmi33github/Porto_AI_Streamlit/main/fine-tuning_distilbert_for_NER_in_restaurant_search/train.bio.txt", sep="\t", header=None)
test = pd.read_csv("https://raw.githubusercontent.com/Syahmi33github/Porto_AI_Streamlit/main/fine-tuning_distilbert_for_NER_in_restaurant_search/test.bio.txt", sep="\t", header=None)

print("train shape:", train.shape)
print("test shape:", test.shape)
print(train.head())
print(test.head())

response = requests.get("https://raw.githubusercontent.com/Syahmi33github/Porto_AI_Streamlit/main/fine-tuning_distilbert_for_NER_in_restaurant_search/train.bio.txt")
response = response.text.splitlines()
print("Response:", response[:5])  # Print the first 5 lines of the response
print("Response length:", len(response))

train_tokens = []
train_tags = []
temp_tokens = []
temp_tags = []
for line in response:
    if line != "":
        tag, token = line.strip().split("\t")
        temp_tags.append(tag)
        temp_tokens.append(token)
    else:
        train_tokens.append(temp_tokens)
        train_tags.append(temp_tags)
        temp_tokens, temp_tags = [], []

print("train_tokens:", train_tokens[:5])
print("train_tags:", train_tags[:5])