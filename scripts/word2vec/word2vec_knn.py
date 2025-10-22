import numpy as np
import pandas as pd

DIR = './data/word_embeddings/word2vec_embeddings'
vectors_path = f'{DIR}/token_vectors_100d.npy'
tokens_path = f'{DIR}/token_analysis_data.csv'

vectors_matrix = np.load(vectors_path)  
tokens_df = pd.read_csv(tokens_path)    

tokens_list = tokens_df['token'].tolist()
vectors_norm = vectors_matrix / np.linalg.norm(vectors_matrix, axis=1, keepdims=True)

word_embeddings = {token: vectors_matrix[i] for i, token in enumerate(tokens_list)}

def cosine_similarity(vec, vecs):
    vec = vec / np.linalg.norm(vec)
    return np.dot(vecs, vec)

def get_top_similar(token, topn=5):
    if token not in word_embeddings:
        return []
    vec = word_embeddings[token]
    sim = cosine_similarity(vec, vectors_norm)
    idx = tokens_list.index(token)
    best_idx = np.argsort(sim)[::-1]
    best_idx = best_idx[best_idx != idx]  
    return [(tokens_list[i], sim[i]) for i in best_idx[:topn]]

def print_knn_report(tokens_list_input, topn=5, label=None):
    if label:
        print(f"\nTop {len(tokens_list_input)} słów dla etykiety '{label}'")
    else:
        print(f"\nTop {len(tokens_list_input)} najczęstszych słów w całym korpusie")
    
    for token in tokens_list_input:
        print(f"\nSłowo: {token}")
        top_sim = get_top_similar(token, topn=topn)
        if top_sim:
            for w, score in top_sim:
                print(f"  {w} -> {score:.4f}")
        else:
            print("  [Brak wektora w modelu]")


top20_tokens = tokens_df.sort_values('Total_Count', ascending=False).head(20)['token'].tolist()
top10_hate = tokens_df.sort_values('Hate_Ratio', ascending=False).head(10)['token'].tolist()
top10_normal = tokens_df.sort_values('Hate_Ratio', ascending=True).head(10)['token'].tolist()


print_knn_report(top20_tokens, topn=5)
print_knn_report(top10_hate, topn=5, label='contains harmful')
print_knn_report(top10_normal, topn=5, label='normal')

SAVE_PATH = f'./outputs/word2vec_results.txt'
with open(SAVE_PATH, 'w', encoding='utf-8') as f:
    import sys
    sys.stdout = f
    print_knn_report(top20_tokens, topn=5)
    print_knn_report(top10_hate, topn=5, label='contains harmful')
    print_knn_report(top10_normal, topn=5, label='normal')
    sys.stdout = sys.__stdout__

print(f"\nRaport zapisany w: {SAVE_PATH}")