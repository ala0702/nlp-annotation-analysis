from sklearn.metrics import cohen_kappa_score

import pandas as pd
import sys

if len(sys.argv)>1:
    iteration = sys.argv[1]
else:
    iteration = "anotated_dataset"


#użyłem do tego GPT
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from collections import Counter
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from collections import Counter
import numpy as np
import re

def fleiss_kappa(matrix):
    """
    Computes Fleiss' kappa for inter-rater agreement.
    matrix: 2D list or numpy array (rows=samples, cols=categories) with counts per category.
    """
    N, k = matrix.shape
    n = np.sum(matrix[0])
    p = np.sum(matrix, axis=0) / (N * n)
    P = (np.sum(matrix**2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P)
    P_e_bar = np.sum(p**2)
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def analyze_dataframe(df: pd.DataFrame):
    """
    Performs several analyses on the dataframe:
    1. Transforms 'label' -> 'label_transformed' ([normal] -> 0, [contains harmful] -> 1)
    2. Counts the occurrences of 0 and 1 in both 'label_transformed' and 'labels'
    3. Calculates Cohen's and Fleiss' kappa between 'label_transformed' and 'labels'
    4. Calculates average text length (characters and words)
    5. Finds the most common non-anonymized word
    6. Prints confusion matrix and summary nicely
    """

    # --- Label transformation ---
    def map_label(label):
        if isinstance(label, list):
            value = label[0]
        elif isinstance(label, str):
            value = label.strip("[]").strip("'\"")
        else:
            return None
        
        if isinstance(value, int):
            return value
        if "normal" in str(value).lower():
            return 0
        elif "contains harmful" in str(value).lower():
            return 1
        else:
            return None

    df["label_transformed"] = df["label"].apply(map_label)
    df["labels"] = df["labels"].apply(map_label)

    # Drop NaN rows (if any) before agreement metrics
    df = df.dropna(subset=["label_transformed", "labels"])

    # --- Counts ---
    label_counts = df["label_transformed"].value_counts().to_dict()
    labels_counts = df["labels"].value_counts().to_dict()

    # --- Cohen’s kappa ---
    try:
        cohens_kappa = cohen_kappa_score(df["label_transformed"], df["labels"])
    except Exception:
        cohens_kappa = None

    # --- Fleiss' kappa ---
    # Build N x k matrix (each row: one item, each column: count of raters per category)
    data_for_fleiss = []
    for _, row in df.iterrows():
        counts = [0, 0]  # 2 categories (0 and 1)
        counts[int(row["label_transformed"])] += 1
        counts[int(row["labels"])] += 1
        data_for_fleiss.append(counts)
    data_for_fleiss = np.array(data_for_fleiss)
    fleiss = fleiss_kappa(data_for_fleiss)

    # --- Confusion matrix ---
    cm = confusion_matrix(df["labels"], df["label_transformed"], labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])

    # --- Average text length ---
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].apply(lambda x: len(str(x).split()))
    avg_char_len = df["char_len"].mean()
    avg_word_len = df["word_len"].mean()

    # --- Most common non-anonymized word ---
    all_words = re.findall(r"\b\w+\b", " ".join(df["text"].astype(str)).lower())
    filtered_words = [w for w in all_words if w != "anonymized_account"]
    
    most_common_word, count = Counter(filtered_words).most_common(1)[0] if filtered_words else (None, 0)

    # --- Print nicely ---
    print("\n===== Label Counts =====")
    print(f"label_transformed: {label_counts}")
    print(f"labels:            {labels_counts}")

    print("\n===== Agreement Metrics =====")
    print(f"Cohen’s kappa: {cohens_kappa:.4f}")
    print(f"Fleiss’ kappa: {fleiss:.4f}")

    print("\n===== Confusion Matrix =====")
    print(cm_df)

    print("\n===== Text Statistics =====")
    print(f"Average character length: {avg_char_len:.2f}")
    print(f"Average word length:      {avg_word_len:.2f}")
    print(f"Most common word:         '{most_common_word}' ({count} times)")
    print(f"unique words: {len(dict.fromkeys(filtered_words))}")
    # --- Return summary dict ---
    results = {
        "label_transformed_counts": label_counts,
        "labels_counts": labels_counts,
        "cohens_kappa": cohens_kappa,
        "fleiss_kappa": fleiss,
        "confusion_matrix": cm_df,
        "avg_char_length": avg_char_len,
        "avg_word_length": avg_word_len,
        "most_common_word": most_common_word,
        "most_common_word_count": count
    }

    return results




dataAlicja = pd.read_json(f'../data/{iteration}/alicja.jsonl', lines=True)
dataBartek = pd.read_json(f'../data/{iteration}/bartek.jsonl', lines=True)
dataKuba = pd.read_json(f'../data/{iteration}/kuba.jsonl', lines=True)
dataAll = pd.concat([dataAlicja, dataBartek, dataKuba])

for name in ["alicja","bartek","kuba"]:
    print(f'\n\n\n{name.capitalize()}\'s data')
    df = pd.read_json(f'../data/{iteration}/{name}.jsonl', lines=True)
    analyze_dataframe(df)
    
analyze_dataframe(dataAll)