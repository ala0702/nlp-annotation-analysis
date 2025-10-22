from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

import numpy as np
import os
import pandas as pd


dataAlicja = pd.read_json(f'../data/anotated_dataset/alicja.jsonl', lines=True)
dataBartek = pd.read_json(f'../data/anotated_dataset/bartek.jsonl', lines=True)
dataKuba = pd.read_json(f'../data/anotated_dataset/kuba.jsonl', lines=True)

dataAll = pd.concat([dataAlicja, dataBartek, dataKuba])

dataAll.drop('Comments', axis=1, inplace=True)

with open('../data./stopwords/polish.txt', 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f)
    stopwords |= {'anonymized_account', '@anonymized_account'}

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text, language='polish')
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

dataAll['tokens'] = dataAll['text'].apply(preprocess)

all_tokens = dataAll['tokens'].tolist()
unique_tokens = set(token for sublist in all_tokens for token in sublist)

print(f"Liczba unikalnych tokenów w 1500 tekstach: {len(unique_tokens)}")

model_path = '../models/nkjp+wiki-forms-all-100-skipg-ns.txt.gz'
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)

print(f"\nŁadowanie modelu Word2Vec zakończone.")

word_embeddings = {}
oov_count = 0

for token in unique_tokens:
    if token in word_vectors:
        word_embeddings[token] = word_vectors.get_vector(token)
    else:
        oov_count += 1

print(f"\nGenerowanie Word Embedding zakończone.")
print(f"Liczba tokenów z wektorami (Word Embeddings): {len(word_embeddings)}")
print(f"Liczba tokenów OOV (bez wektora): {oov_count}")

tokens_list = list(word_embeddings.keys())
vectors_matrix = np.array(list(word_embeddings.values()))

# Wymiar macierzy: (Liczba tokenów, 100)
print(f"Macierz wektorów do zapisu ma wymiar: {vectors_matrix.shape}")

SAVE_DIR = '../data/word_embeddings/word2vec_embeddings'
os.makedirs(SAVE_DIR, exist_ok=True)

# macierz wektorów
vectors_path = os.path.join(SAVE_DIR, 'token_vectors_100d.npy')
np.save(vectors_path, vectors_matrix)

# lista tokenów
tokens_df = pd.DataFrame(tokens_list, columns=['token'])
tokens_path = os.path.join(SAVE_DIR, 'token_dictionary.csv')
tokens_df.to_csv(tokens_path, index=False)

