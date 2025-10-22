from collections import defaultdict
from gensim.models import KeyedVectors

import numpy as np
import os
import pandas as pd
import re


dataAlicja = pd.read_json(f'./data/anotated_dataset/alicja.jsonl', lines=True)
dataBartek = pd.read_json(f'./data/anotated_dataset/bartek.jsonl', lines=True)
dataKuba = pd.read_json(f'./data/anotated_dataset/kuba.jsonl', lines=True)

dataAll = pd.concat([dataAlicja, dataBartek, dataKuba])
dataAll.drop('Comments', axis=1, inplace=True)

with open('./data./stopwords/polish.txt', 'r', encoding='utf-8') as f:
    stopwords = set(line.strip().lower() for line in f if line.strip())
    stopwords |= {'anonymized_account', '@anonymized_account'}

def preprocess(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in stopwords and len(t) > 2 and not t.isdigit()]

dataAll['tokens'] = dataAll['text'].apply(preprocess)

all_tokens = dataAll['tokens'].tolist()
unique_tokens = set(token for sublist in all_tokens for token in sublist)

print(f"Liczba unikalnych tokenów w 1500 tekstach: {len(unique_tokens)}")





token_counts = defaultdict(lambda: {'total': 0, 'hate': 0})
for index, row in dataAll.iterrows():
    label = row['label']
    tokens = row['tokens']

    is_hate = 'contains harmful' in label

    for token in set(tokens):
        token_counts[token]['total'] += 1
        if is_hate:
            token_counts[token]['hate'] += 1

hate_stats_list = []
for token, counts in token_counts.items():
    if counts['total'] > 0:
        ratio = counts['hate'] / counts['total']
        hate_stats_list.append({'token': token, 'Total_Count': counts['total'], 'Hate_Ratio': ratio})
    
hate_stats_df = pd.DataFrame(hate_stats_list)





model_path = './models/nkjp+wiki-forms-all-100-skipg-ns.txt.gz'
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)

print(f"\nŁadowanie modelu Word2Vec zakończone.")

word_embeddings = {}
oov_count = 0

for token in unique_tokens:
    if token in word_vectors:
        word_embeddings[token] = word_vectors.get_vector(token)
    else:
        oov_count += 1

tokens_list = list(word_embeddings.keys())
vectors_matrix = np.array(list(word_embeddings.values()))

print(f"Liczba tokenów z wektorami (Word Embeddings): {len(word_embeddings)}")
print(f"Liczba tokenów OOV (bez wektora): {oov_count}")

# Wymiar macierzy: (Liczba tokenów, 100)
print(f"Macierz wektorów do zapisu ma wymiar: {vectors_matrix.shape}")

SAVE_DIR = './data/word_embeddings/word2vec_embeddings'
os.makedirs(SAVE_DIR, exist_ok=True)

# macierz wektorów
vectors_path = os.path.join(SAVE_DIR, 'token_vectors_100d.npy')
np.save(vectors_path, vectors_matrix)

# lista tokenów
tokens_df = pd.DataFrame(tokens_list, columns=['token'])

final_tokens_df = pd.merge(tokens_df, hate_stats_df, on = 'token', how = 'left')
final_tokens_df.fillna({'Total_Count': 0, 'Hate_Ratio': 0}, inplace=True)

tokens_path = os.path.join(SAVE_DIR, 'token_analysis_data.csv')
final_tokens_df.to_csv(tokens_path, index=False)

