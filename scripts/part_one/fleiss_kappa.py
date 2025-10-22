from statsmodels.stats.inter_rater import fleiss_kappa

import numpy as np
import pandas as pd
import sys

iteration = sys.argv[1]

dataAlicja = pd.read_json(f'../../data/{iteration}/alicja.jsonl', lines=True)['label'].apply(lambda x: x[0]).to_numpy()
dataBartek = pd.read_json(f'../../data/{iteration}/bartek.jsonl', lines=True)['label'].apply(lambda x: x[0]).to_numpy()
dataKuba = pd.read_json(f'../../data/{iteration}/kuba.jsonl', lines=True)['label'].apply(lambda x: x[0]).to_numpy()

data = np.array([dataAlicja, dataBartek, dataKuba]).T


categories = ['normal', 'contains harmful']
number_of_objects = data.shape[0]
number_of_categories = len(categories)

count_matrix = np.zeros((number_of_objects, number_of_categories), dtype=int)

for i in range(number_of_objects):
    for j in range(number_of_categories):
        count_matrix[i, j] = np.sum(data[i] == categories[j])

kappa = fleiss_kappa(count_matrix)

print(f"Fleiss's Kappa: {kappa}")

if kappa < 0:
    interpretation = "Słaba zgodność"
elif kappa < 0.20:
    interpretation = "Śladowa zgodność"
elif kappa < 0.40:
    interpretation = "Lekka zgodność"
elif kappa < 0.60:
    interpretation = "Umiarkowana zgodność"
elif kappa < 0.80:
    interpretation = "Istotna zgodność"
else:
    interpretation = "Niemal perfekcyjna zgodność"

print(f"Interpretacja: {interpretation}")
