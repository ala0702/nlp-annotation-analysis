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


X_words = np.array(list(word_vectors.values()))
words_list = list(word_vectors.keys())

tsne_words = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_2d_words = tsne_words.fit_transform(X_words)

fig_words = px.scatter(
    x=X_2d_words[:,0],
    y=X_2d_words[:,1],
    text=words_list,
    title="t-SNE FastText - Embedding słów"
)
fig_words.update_traces(textposition='top center')
fig_words.show()
fig_words.write_html("./outputs/tsne_fasttext_words.html")

all_tokens = []
for t in texts:
    all_tokens.extend(tokenize(t))

freq_words = [w for w, _ in Counter(all_tokens).most_common(20)]
print("\nTop 20 najczęstszych słów w całym korpusie")
for w in freq_words:
    if w in wv:
        similar = wv.most_similar(w, topn=5)
        print(f"\nSłowo: {w}")
        for sim_word, score in similar:
            print(f"  {sim_word} -> {score:.4f}")

labels_set = set(labels)
for label in labels_set:
    tokens_label = []
    for text, l in zip(texts, labels):
        if l == label:
            tokens_label.extend(tokenize(text))
    freq_label_words = [w for w, _ in Counter(tokens_label).most_common(10)]
    
    print(f"\nTop 10 słów dla etykiety '{label}'")
    for w in freq_label_words:
        if w in wv:
            similar = wv.most_similar(w, topn=5)
            print(f"\nSłowo: {w}")
            for sim_word, score in similar:
                print(f"  {sim_word} -> {score:.4f}")


results = {
    "global_top_words": {},
    "labels_top_words": {}
}

# --- Global top words ---
for w in freq_words:
    if w in wv:
        results["global_top_words"][w] = wv.most_similar(w, topn=5)

# --- Label-specific top words ---
for label in labels_set:
    tokens_label = []
    for text, l in zip(texts, labels):
        if l == label:
            tokens_label.extend(tokenize(text))
    freq_label_words = [w for w, _ in Counter(tokens_label).most_common(10)]
    
    label_results = {}
    for w in freq_label_words:
        if w in wv:
            label_results[w] = wv.most_similar(w, topn=5)
    results["labels_top_words"][label] = label_results

# Save results as JSON and text
with open("./outputs/fasttext_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open("./outputs/fasttext_results.txt", "w", encoding="utf-8") as f:
    f.write("Top 20 najczęstszych słów w całym korpusie\n\n")
    for w, sims in results["global_top_words"].items():
        f.write(f"Słowo: {w}\n")
        for sim_word, score in sims:
            f.write(f"  {sim_word} -> {score:.4f}\n")
        f.write("\n")
    for label, words in results["labels_top_words"].items():
        f.write(f"\nTop 10 słów dla etykiety '{label}'\n")
        for w, sims in words.items():
            f.write(f"Słowo: {w}\n")
            for sim_word, score in sims:
                f.write(f"  {sim_word} -> {score:.4f}\n")
            f.write("\n")


summary = {
    "total_comments": len(texts),
    "unique_words": len(words),
    "words_in_model": len(word_vectors),
    "words_missing_in_model": len(words) - len(word_vectors),
    "labels": list(labels_set)
}

with open("./outputs/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
