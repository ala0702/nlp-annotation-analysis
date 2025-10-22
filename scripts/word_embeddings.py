from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt_tab')

import pandas as pd


dataAlicja = pd.read_json(f'../data/anotated_dataset/alicja.jsonl', lines=True)
dataBartek = pd.read_json(f'../data/anotated_dataset/bartek.jsonl', lines=True)
dataKuba = pd.read_json(f'../data/anotated_dataset/kuba.jsonl', lines=True)

dataAll = pd.concat([dataAlicja, dataBartek, dataKuba])
# print(f'Total examples in dataset: {len(dataAll)}')

dataAll.drop('Comments', axis=1, inplace=True)
# print(dataAll.head())

with open('../data./stopwords/polish.txt', 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)

# print(len(stopwords))

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text, language='polish')
    tokens = [word for word in tokens if word not in stopwords and word != 'anonymized_account' and word.isalpha()]
    return tokens

dataAll['tokens'] = dataAll['text'].apply(preprocess)

pd.set_option('display.max_colwidth', None) 
print(dataAll[['text', 'tokens']].head(10))
