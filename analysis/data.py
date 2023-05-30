import os
import numpy as np
import pandas as pd
import gdown
import subprocess

CACHE_DIR = '/dfs/scratch0/merty/.cache/pMHC/'
CLASSIFICATION_DATA_GDRIVE_ID = "1SKoEGuJjse5Ta-WZr6fht96iOAMoZDAB"

def get_classification_data(): # TODO: Where's training/valid/test?
    classification_path = os.path.join(CACHE_DIR, "classification.zip")
    if not os.path.exists(classification_path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        subprocess.call(["gdown", "--id", CLASSIFICATION_DATA_GDRIVE_ID, "--output", classification_path])
        subprocess.call(["unzip", "-qq", classification_path, "-d", CACHE_DIR])
    
    df = pd.read_csv(os.path.join(CACHE_DIR, "sage-haze-125_ckpt_e37_i16067_clssfctnData.csv"))
    latent_cols = [col for col in df.columns if col.startswith("latent_")]
    
    test_embeddings = np.zeros((df.shape[0], len(latent_cols)))
    for col in latent_cols:
        test_embeddings[:, int(col.split("_")[-1])] = df[col].values
    test_labels = df["affinity"].values
    test_logits = df["pred_affinity"].values
    test_predictions = (test_logits > 0.426).astype(int) # What is this number? Took from generate_embeddings.py
    return test_embeddings, test_logits, test_predictions, test_labels

test_embeddings, test_logits, test_predictions, test_labels = get_classification_data()