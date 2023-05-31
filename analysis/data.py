import os
import numpy as np
import pandas as pd
import gdown
import subprocess
from sklearn.model_selection import train_test_split

CACHE_DIR = '/dfs/scratch0/merty/.cache/pMHC/'
CLASSIFICATION_DATA_GDRIVE_ID = "1SKoEGuJjse5Ta-WZr6fht96iOAMoZDAB"
REGRESSION_DATA_GDRIVE_ID = "1aoX5Jv1uSCQav0dzq9eeA4lDaV5ZCPrZ"

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


def get_all_data(load_train_pct=1.0, seed=1):
    """
    :param load_train_pct: What percentage of the training data to subsample. We stratify by the training labels.
    :param seed:
    :return:
    """
    classification_path = os.path.join(CACHE_DIR, "classification.zip")
    if not os.path.exists(classification_path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        subprocess.call(["gdown", "--id", CLASSIFICATION_DATA_GDRIVE_ID, "--output", classification_path])
        subprocess.call(["unzip", "-qq", classification_path, "-d", CACHE_DIR])


    df = pd.read_csv(os.path.join(CACHE_DIR, "sage-haze-125_ckpt_e37_i16067_clssfctnData.csv"))
    latent_cols = [col for col in df.columns if col.startswith("latent_")]
    
    train_embeddings = np.zeros((df.shape[0], len(latent_cols)))
    for col in latent_cols:
        train_embeddings[:, int(col.split("_")[-1])] = df[col].values
    train_labels = df["affinity"].values
    train_logits = df["pred_affinity"].values
    train_predictions = (train_logits > 0.426).astype(int)  # What is this number? Took from generate_embeddings.py

    if load_train_pct < 1:
        train_embeddings, _, train_labels, _, train_logits, _, train_predictions, _= train_test_split(train_embeddings,
                                                                                                      train_labels,
                                                                                                      train_logits,
                                                                                                      train_predictions,
                                                                                                      train_size=load_train_pct,
                                                                                                      random_state=seed,
                                                                                                      stratify=train_labels)

    regression_path = os.path.join(CACHE_DIR, "regression.zip")
    if not os.path.exists(regression_path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        subprocess.call(["gdown", "--id", REGRESSION_DATA_GDRIVE_ID, "--output", regression_path])
        subprocess.call(["unzip", "-qq", regression_path, "-d", CACHE_DIR])

    df = pd.read_csv(os.path.join(CACHE_DIR, "sage-haze-125_ckpt_e37_i16067_rgrssnData.csv"))
    latent_cols = [col for col in df.columns if col.startswith("latent_")]
    test_embeddings = np.zeros((df.shape[0], len(latent_cols)))
    for col in latent_cols:
        test_embeddings[:, int(col.split("_")[-1])] = df[col].values
    test_labels = (df["affinity"].values > 0.426).astype(int) ## Again, are we sure?
    test_logits = df["pred_affinity"].values
    test_predictions = (test_logits > 0.426).astype(int)  # What is this number? Took from generate_embeddings.py
    return train_embeddings, train_logits, train_predictions, train_labels, test_embeddings, test_logits, test_predictions, test_labels

train_embeddings, train_logits, train_predictions, train_labels, test_embeddings, test_logits, test_predictions, test_labels = get_all_data()