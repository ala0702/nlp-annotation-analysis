from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import altair as alt
import numpy as np
import pandas as pd
import os


DIR = './data/word_embeddings/word2vec_embeddings'
vectors_path = os.path.join(DIR, 'token_vectors_100d.npy')
tokens_path = os.path.join(DIR, 'token_analysis_data.csv') 

X_tokens = np.load(vectors_path)
tokens_df = pd.read_csv(tokens_path) 

print('Wczytano dane. Rozmiar macierzy wektorów:', X_tokens.shape)
print('t-SNE started')

X_pca = PCA(n_components=50, random_state=42).fit_transform(X_tokens)

tsne = TSNE(n_components=2, 
            random_state=42, 
            perplexity=50, 
            n_iter=500, 
            init='pca', 
            learning_rate='auto')
X_2d = tsne.fit_transform(X_pca)

print('t-SNE finished')

df_tsne = pd.DataFrame(X_2d, columns=['TSNE-1', 'TSNE-2'])
df_tsne['token'] = tokens_df['token'] 

df_final = pd.merge(df_tsne, tokens_df, on='token', how='inner')

print(f"Dane do wizualizacji gotowe: {len(df_final)} tokenów.")

alt.data_transformers.disable_max_rows()
tooltip_list = [
    'token', 
    alt.Tooltip('Hate_Ratio', format='.3f', title='Hate Ratio'), 
    alt.Tooltip('Total_Count', title='Wystąpienia')
]

chart = alt.Chart(df_final).mark_circle(size=60, opacity=0.8).encode(
    x=alt.X('TSNE-1', title='t-SNE Wymiar 1'),
    y=alt.Y('TSNE-2', title='t-SNE Wymiar 2'),
    
    color=alt.Color('Hate_Ratio', 
                    scale=alt.Scale(range='heatmap', domain=[0, 1]), 
                    title='Współczynnik Nienawiści'), 
    
    size=alt.Size('Total_Count', 
                  scale=alt.Scale(range=[10, 300], domain=[df_final['Total_Count'].min(), df_final['Total_Count'].max()]),
                  title='Całkowita Liczba Wystąpień'),
    
    tooltip=tooltip_list
).properties(
    title=f"t-SNE Osadzeń Słów (Word Embeddings) - 100D Skip-gram"
).interactive() 

chart.save('./plots/word2vec_tsne_interactive.json')
