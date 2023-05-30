import os.path as osp
import copy
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from model_lib.initialise_model import initialise_model
from dataset import get_dataloader
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
from vocab import * 


def log_wandb(model_output, true_labels, loss, folder='train'):

    """ 
    Log metrics to wandb. 
    Will always log loss, but only logs AUC / PRC if 
    postive class is present.
    
    Args:
        model_output (torch.Tensor): The output tensor from the model predicting binding affinity.
        true_labels (torch.Tensor): The true labels (binding affinities).
        loss (torch.Tensor): The BCE loss value to log.
        folder (str, optional): The folder name to log the data in wadnb. Defaults to 'train'.

    Returns:
        None
    """
    sigmoid = nn.Sigmoid()
    model_output = sigmoid(model_output).detach().cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()

    BA_true_mean = true_labels.mean()
    BA_pred_std = model_output.std()

    metrics = {f"{folder}/loss": loss.item(), f'{folder}/BA_mean': BA_true_mean, f'{folder}/BA_std': BA_pred_std}

    if len(np.unique(true_labels)) == 2: 
        # only compute AUROC / AUPRC if postive class is present
        auroc = roc_auc_score(true_labels, model_output)
        auprc = average_precision_score(true_labels, model_output)
        metrics.update({f'{folder}/aucroc': auroc, f'{folder}/aucprc': auprc})
    
    wandb.log(metrics)


def load_pretrained(args, model_path, device):
    model = initialise_model(args, vocab_size=VOCAB_SIZE, num_classes=1, device=device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


def reweight_dataloader(args, model, device, sel_type='topr'):
    # Load data - training data is the MA classifcation data. We ignore the 8ish million
    # multiple allele data and keep the 4ish million single allele data
    unshuffled_train_loader, train_ds = get_dataloader(args.tr_df_path, cv_splits = None, 
                                             peptide_repr = args.peptide_repr, 
                                             mhc_repr = args.mhc_repr,
                                             batch_size = args.batch_size,
                                             shuffle = False, return_df=True)
    ##############################################
    #               Get predictions              # 
    ##############################################
    model.eval()
    with torch.no_grad():
        probs = []
        all_affinity = []
        for data in unshuffled_train_loader:
            sigmoid = torch.nn.Sigmoid()
            peptide = data['peptide'].long().to(device)
            mhc = data['mhc'].long().to(device)
            affinity = data['BA'].long().to(device)
            pred_affinity = sigmoid(model(peptide, mhc))
            prob = torch.zeros(pred_affinity.size(0), 2).to(device)
            prob[:, 0] = pred_affinity
            prob[:, 1] = 1 - pred_affinity
            probs.append(prob[:, affinity].size(-1))
            all_affinity.append(affinity)
        
        probs = torch.concat(probs)
        all_affinity = torch.concat(all_affinity)
        if sel_type == 'topr':
            # select top r% by each class
            selection_id = []
            for cls in torch.unique(all_affinity):
                per_cls_rank = torch.argsort(-probs[all_affinity == cls])
                per_cls_rank = per_cls_rank[:int(len(per_cls_rank) * args.threshold)]
                selection_id.append(torch.arange(len(probs))[all_affinity == cls][per_cls_rank])
            selection_id = torch.concat(selection_id)
        elif sel_type == 'value':
            selection = probs > args.threshold
            selection_id = torch.arange(len(selection))[selection]
        else:
            raise NotImplemented
    
    selected_df = train_ds[selection_id]
    train_loader = DataLoader(selected_df, batch_size=args.batch_size, shuffle=True)
    return train_loader