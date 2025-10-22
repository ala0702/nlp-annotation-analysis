import json
import os
import pandas as pd

iteration1 = pd.read_json(f'../../data/iteration1_annotations/sample1_doccano.jsonl', lines=True)
iteration2 = pd.read_json(f'../../data/iteration2_annotations/sample2_doccano.jsonl', lines=True)
full_dataset = pd.read_json(f'../../data/full_dataset_annotations/full_dataset_doccano.jsonl', lines=True)

texts_to_drop = pd.concat([iteration1['text'], iteration2['text']]).unique()
indices_to_drop = full_dataset[full_dataset['text'].isin(texts_to_drop)].index
full_dataset = full_dataset.drop(indices_to_drop)

sample_size = 500
full_dataset = full_dataset.sample(frac=1).reset_index(drop=True)
subset_ala = full_dataset.sample(n=sample_size, random_state=42) 
full_dataset_temp = full_dataset.drop(subset_ala.index)
subset_bartek = full_dataset_temp.sample(n=sample_size, random_state=42)
subset_kuba_remaining = full_dataset_temp.drop(subset_bartek.index)
subset_kuba = subset_kuba_remaining.sample(n=sample_size, random_state=42)

subset_ala_file = os.path.join('../data/full_dataset_annotations/subset_ala.jsonl')
with open(subset_ala_file, 'w', encoding='utf-8') as f:
    records = subset_ala.to_dict(orient='records')
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

subset_bartek_file = os.path.join('../data/full_dataset_annotations/subset_bartek.jsonl')
with open(subset_bartek_file, 'w', encoding='utf-8') as f:
    records = subset_bartek.to_dict(orient='records')
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

subset_kuba_file = os.path.join('../data/full_dataset_annotations/subset_kuba.jsonl')
with open(subset_kuba_file, 'w', encoding='utf-8') as f:
    records = subset_kuba.to_dict(orient='records')
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

