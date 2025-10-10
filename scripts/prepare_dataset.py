import random
from datasets import load_dataset
import json
import pandas as pd
import os

# -----------------------------
# SETTINGS
# -----------------------------
TASK = "task01"  # Choose task01 (2 classes) or task02 (3 classes)
ITER1_SIZE = 100  # Number of comments to sample for iteration 1
ITER2_SIZE = 100  # Number of comments to sample for iteration 2
RANDOM_SEED = 42  # For reproducibility
OUTPUT_DIR = "./data"
OUTPUT_DIR_ITER1 = os.path.join(OUTPUT_DIR, "iteration1_annotations")
OUTPUT_DIR_ITER2 = os.path.join(OUTPUT_DIR, "iteration2_annotations")
ANNOTATORS = ["annotator1", "annotator2", "annotator3"]  # List of annotators

# -----------------------------
# SET SEED FOR REPRODUCIBILITY
# -----------------------------
random.seed(RANDOM_SEED)

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = load_dataset(
    "poleval/poleval2019_cyberbullying",
    "task01",              # Choose task01 (2 classes) or task02 (3 classes)
    trust_remote_code=True
)
data = dataset['train']
df = pd.DataFrame(data)

print(f"Total examples in dataset: {len(data)}")
print(df.columns)
print(df.head())

# -----------------------------
# FUNCTION TO SAVE SAMPLE TO JSONL
# -----------------------------
def save_sample_to_jsonl(sample_df, folder_path, file_name):
    """
    Saves a pandas DataFrame to a Doccano-compatible JSONL file.

    Args:
        sample_df (pd.DataFrame): Sampled dataset (must have 'text' and 'label').
        folder_path (str): Folder where JSONL will be saved.
        file_name (str): Name of the JSONL file.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create full file path
    output_file = os.path.join(folder_path, file_name)

    # Convert DataFrame rows to Doccano JSONL format
    doccano_data = []
    for _, row in sample_df.iterrows():
        doccano_data.append({
            "text": row['text'],
            "labels": [row['label']]  # Doccano expects a list of labels
        })

    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in doccano_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Saved {len(sample_df)} examples to {output_file}")


# -----------------------------
# ITERATION 1
# -----------------------------
sample1_df = df.sample(n=ITER1_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
save_sample_to_jsonl(sample1_df, OUTPUT_DIR_ITER1, "sample1_doccano.jsonl")

# -----------------------------
# ITERATION 2 (without first sample)
# -----------------------------
remaining_df = df.drop(sample1_df.index)
sample2_df = remaining_df.sample(n=ITER2_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
save_sample_to_jsonl(sample2_df, OUTPUT_DIR_ITER2, "sample2_doccano.jsonl")

# -----------------------------
# OPTIONAL: Save FULL DATA for final annotation
# -----------------------------
FULL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "full_dataset_annotations")
full_sample_df = df  # Can filter/remove previous iteration samples if needed
save_sample_to_jsonl(full_sample_df, FULL_OUTPUT_DIR, "full_dataset_doccano.jsonl")