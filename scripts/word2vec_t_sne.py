from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import altair as alt
import numpy as np
import pandas as pd
import os

# ----------------------------------------------------------------------
# Ścieżki i Ładowanie Danych
# ----------------------------------------------------------------------
DIR = '../data/word_embeddings/word2vec_embeddings'
vectors_path = os.path.join(DIR, 'token_vectors_100d.npy')
# Używamy ujednoliconego pliku z danymi analitycznymi
tokens_path = os.path.join(DIR, 'token_analysis_data.csv') 

# Wczytywanie macierzy wektorów i danych analitycznych
X_tokens = np.load(vectors_path)
tokens_df = pd.read_csv(tokens_path) 
# tokens_list = tokens_df['token'].tolist() # Niepotrzebne, ponieważ użyjemy tokens_df

print('Wczytano dane. Rozmiar macierzy wektorów:', X_tokens.shape)

# ----------------------------------------------------------------------
# Redukcja Wymiarowości (t-SNE)
# ----------------------------------------------------------------------
print('t-SNE started')

# PCA do 50 wymiarów (Dobra praktyka inicjalizacji t-SNE)
X_pca = PCA(n_components=50, random_state=42).fit_transform(X_tokens)

# t-SNE do 2 wymiarów
tsne = TSNE(n_components=2, 
            random_state=42, 
            perplexity=50, 
            n_iter=500, 
            init='pca', 
            learning_rate='auto')
X_2d = tsne.fit_transform(X_pca)

print('t-SNE finished')

# ----------------------------------------------------------------------
# 1. INTEGRACJA DANYCH
# ----------------------------------------------------------------------

# Konwersja wyników t-SNE na DataFrame
df_tsne = pd.DataFrame(X_2d, columns=['TSNE-1', 'TSNE-2'])
# Dodanie kolumny tokenów (musimy założyć, że kolejność wierszy jest identyczna jak w X_tokens)
df_tsne['token'] = tokens_df['token'] 

# Połączenie wyników t-SNE z danymi anotacyjnymi (Hate_Ratio, Total_Count)
df_final = pd.merge(df_tsne, tokens_df, on='token', how='inner')

print(f"Dane do wizualizacji gotowe: {len(df_final)} tokenów.")

# ----------------------------------------------------------------------
# 2. GENEROWANIE WYKRESU INTERAKTYWNEGO (Altair)
# ----------------------------------------------------------------------

# Ustawienie opcji Altair na większy limit danych
alt.data_transformers.disable_max_rows()

# Definiowanie elementów dymka (tooltip)
tooltip_list = [
    'token', 
    alt.Tooltip('Hate_Ratio', format='.3f', title='Hate Ratio'), 
    alt.Tooltip('Total_Count', title='Wystąpienia')
]

chart = alt.Chart(df_final).mark_circle(size=60, opacity=0.8).encode(
    # Pozycje na wykresie
    x=alt.X('TSNE-1', title='t-SNE Wymiar 1'),
    y=alt.Y('TSNE-2', title='t-SNE Wymiar 2'),
    
    # Kolorowanie: Powiązanie z Anotacją
    color=alt.Color('Hate_Ratio', 
                    scale=alt.Scale(range='heatmap', domain=[0, 1]), # Użycie skali ciepła 0-1
                    title='Współczynnik Nienawiści'), 
    
    # Rozmiar punktów (opcjonalnie: większe dla częstszych słów)
    size=alt.Size('Total_Count', 
                  scale=alt.Scale(range=[10, 300], domain=[df_final['Total_Count'].min(), df_final['Total_Count'].max()]),
                  title='Całkowita Liczba Wystąpień'),
    
    # Interaktywny dymek
    tooltip=tooltip_list
).properties(
    title=f"t-SNE Osadzeń Słów (Word Embeddings) - 100D Skip-gram"
).interactive() # Umożliwia zbliżanie i przesuwanie

# ----------------------------------------------------------------------
# 3. ZAPIS WYNIKU
# ----------------------------------------------------------------------

chart.save('../plots/word2vec_tsne_interactive.json')
