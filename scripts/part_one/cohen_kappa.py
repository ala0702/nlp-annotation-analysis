from sklearn.metrics import cohen_kappa_score

import pandas as pd
import sys

iteration = sys.argv[1]

dataAlicja = pd.read_json(f'../../data/{iteration}/alicja.jsonl', lines=True)['label'].apply(lambda x: x[0]).to_numpy()
dataBartek = pd.read_json(f'../../data/{iteration}/bartek.jsonl', lines=True)['label'].apply(lambda x: x[0]).to_numpy()
dataKuba = pd.read_json(f'../../data/{iteration}/kuba.jsonl', lines=True)['label'].apply(lambda x: x[0]).to_numpy()

kappa_alicja_bartek = cohen_kappa_score(dataAlicja, dataBartek)
kappa_alicja_kuba = cohen_kappa_score(dataAlicja, dataKuba)
kappa_bartek_kuba = cohen_kappa_score(dataBartek, dataKuba)

print(f"Cohen's Kappa between Alicja and Bartek: {kappa_alicja_bartek}")
print(f"Cohen's Kappa between Alicja and Kuba: {kappa_alicja_kuba}")
print(f"Cohen's Kappa between Bartek and Kuba: {kappa_bartek_kuba}")    