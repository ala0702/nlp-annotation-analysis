from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import altair as alt
import numpy as np
import pandas as pd
import os
import ast

DIR = './data/word_embeddings/word2vec_embeddings'
vectors_path = os.path.join(DIR, 'token_vectors_100d.npy')
tokens_path = os.path.join(DIR, 'token_analysis_data.csv') 
sentences_path = os.path.join(DIR, 'sentence_token_analysis_data.csv') 

X_tokens = np.load(vectors_path)
tokens_df = pd.read_csv(tokens_path) 
sentences_df = pd.read_csv(sentences_path) 
sentences_df['tokens'] = sentences_df['tokens'].apply(ast.literal_eval)
X_sentence = []
y_sentence = []

token_index = pd.Index(tokens_df['token'])
maxlen = 0
def get_token_idx(token):
    try:
        return token_index.get_loc(token)
    except KeyError:
        return None  # or -1, or np.nan

for id, sentence in sentences_df.iterrows():
    arr = np.zeros(100)
    denom = 0
    for token in sentence["tokens"]:
        idx = get_token_idx(token)
        
        if idx:
            arr += X_tokens[idx]
            denom += 1
    if denom>0:
        if maxlen<denom:
            maxlen=denom
        X_sentence.append(arr/denom)
        y_sentence.append(
            {
                "raw":sentence["raw"],
                "tokens":",".join(sentence["tokens"]),
                "senLength": len(sentence["tokens"]),
                "label":sentence["label"]
            }
            )
    
sentence_y_df = pd.DataFrame(y_sentence)
X_sentence = np.asarray(X_sentence)

print('Wczytano dane. Rozmiar macierzy wektorów:', X_sentence.shape)
print('t-SNE started')

X_pca = PCA(n_components=50, random_state=42).fit_transform(X_sentence)

tsne = TSNE(n_components=2, 
            random_state=42, 
            perplexity=50, 
            # n_iter=500, 
            init='pca', 
            learning_rate='auto')
X_2d = tsne.fit_transform(X_pca)

print('t-SNE finished')

df_tsne = pd.DataFrame(X_2d, columns=['TSNE-1', 'TSNE-2'])
df_tsne['raw'] = sentence_y_df['raw'] 

df_final = pd.merge(df_tsne, sentence_y_df, on='raw', how='inner')
df_final['label_clean'] = df_final['label'].astype(str)

df_final['label_clean'] = df_final['label_clean'].str.strip("[]'").str.replace("'", "")
print(df_final["label_clean"])
print(f"Dane do wizualizacji gotowe: {len(df_final)} tokenów.")

alt.data_transformers.disable_max_rows()
tooltip_list = [
    'raw', 
    alt.Tooltip('senLength', format='.3f', title='Długość zdania'), 
    alt.Tooltip('label_clean', title='oznaczenie')
]

chart = alt.Chart(df_final).mark_circle(size=60, opacity=0.8).encode(
    x=alt.X('TSNE-1', title='t-SNE Wymiar 1'),
    y=alt.Y('TSNE-2', title='t-SNE Wymiar 2'),
    
    # color=alt.Color('senLength', 
    #                 scale=alt.Scale(range='heatmap', domain=[0, maxlen]), 
    #                 title='Długość zdania'), 
    
    color=alt.Color(
        'label_clean',
        scale=alt.Scale(
            domain=["normal", "contains harmful"],
            range=['blue', 'red']  # Different colors for each label
        ),
        title='Etykieta'
    ),
    
    tooltip=tooltip_list
).properties(
    title=f"t-SNE Osadzeń Zdań (Sentence Embeddings) - 100D Skip-gram"
).interactive() 

chart.save('./plots/word2vec_tsne_interactive_sentences_clrs.json')
