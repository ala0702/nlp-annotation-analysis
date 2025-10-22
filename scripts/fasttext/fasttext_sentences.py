import json
import re
import numpy as np
from collections import Counter
from sklearn.manifold import TSNE
import plotly.express as px
from gensim.models.fasttext import load_facebook_model

import os
os.makedirs("outputs", exist_ok=True)

def load_jsonl(file_path):
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            labels.append(data["label"][0] if data["label"] else "unknown")
    return texts, labels

texts1, labels1 = load_jsonl("./data/anotated_dataset/alicja.jsonl")
texts2, labels2 = load_jsonl("./data/anotated_dataset/bartek.jsonl")
texts3, labels3 = load_jsonl("./data/anotated_dataset/kuba.jsonl")

texts = texts1 + texts2 + texts3
labels = labels1 + labels2 + labels3

print(f"Liczba komentarzy: {len(texts)}")


def load_stopwords(filepath="./data/stopwords/polish.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = set(line.strip().lower() for line in f if line.strip())
    # Dodajmy kilka dodatkowych uniwersalnych słów i symboli
    stopwords |= {"anonymized_account", "@anonymized_account"}
    return stopwords

STOPWORDS = load_stopwords()

def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    # Filtracja: usuwamy stopwords, jednoznakowe słowa i liczby
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2 and not t.isdigit()]

words = set()
for t in texts:
    words.update(tokenize(t))

print(f"Liczba unikalnych słów: {len(words)}")

ft_model = load_facebook_model(r"./models/kgr10.plain.skipgram.dim100.neg10.bin")
wv = ft_model.wv

word_vectors = {}
for w in words:
    if w in wv:
        word_vectors[w] = wv[w]

print(f"Liczba słów znalezionych w modelu: {len(word_vectors)}")
print(f"Liczba słów nieznalezionych w modelu: {len(words) - len(word_vectors)}")


X_sentences = []
y_sentences = []
colors = []

for text in texts:
    arr = np.zeros(100)
    tokencount = 0 
    for token in tokenize(text):
        arr += word_vectors[token]
        tokencount+=1
    if tokencount>0:
        X_sentences.append(arr/tokencount)
        y_sentences.append(text)
        colors.append(tokencount)

X_sentences = np.array(X_sentences)


tsne_words = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_2d_sentences = tsne_words.fit_transform(X_sentences)

fig_sentences = px.scatter(
    x=X_2d_sentences[:, 0],
    y=X_2d_sentences[:, 1],
    hover_data={'text': y_sentences},
    color=colors,  # Add color mapping based on sentence length
    color_continuous_scale='viridis',  # Choose a colormap
    title="t-SNE FastText - Sentence Embeddings with Length Coloring"
)

fig_sentences.update_traces(
    textposition='top center',
    marker=dict(size=8)  
)

fig_sentences.update_coloraxes(
    colorbar_title="Sentence Length (tokens)"
)
fig_sentences.update_traces(textposition='top center')
fig_sentences.show()
fig_sentences.write_html("./outputs/tsne_fasttext_sentences.html")