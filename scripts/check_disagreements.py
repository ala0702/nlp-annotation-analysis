import pandas as pd
import sys

iteration = sys.argv[1]

a = pd.read_json(f'../data/{iteration}/alicja.jsonl', lines=True)[['text', 'label']].rename(columns={'label': 'label_alicja'})
b = pd.read_json(f'../data/{iteration}/bartek.jsonl', lines=True)[['text', 'label']].rename(columns={'label': 'label_bartek'})
c = pd.read_json(f'../data/{iteration}/kuba.jsonl', lines=True)[['text', 'label']].rename(columns={'label': 'label_kuba'})

df = a.merge(b, on='text', how='inner').merge(c, on='text', how='inner')

mask_all_equal = (df['label_alicja'] == df['label_bartek']) & (df['label_alicja'] == df['label_kuba'])
df_disagreements = df[~mask_all_equal].reset_index(drop=True)

output_path = sys.argv[2] if len(sys.argv) > 2 else f'../data/{iteration}/disagreements.csv'
df_disagreements.to_csv(output_path, index=False)
print(f'Zapisano do: {output_path}')