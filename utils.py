import os.path as osp
import os
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from model_lib.initialise_model import initialise_model
from dataset import get_dataloader
from vocab import * 
import pdb
from analysis.data import get_all_data



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
    _, train_logits, _, train_labels, _, _, _, _ = get_all_data()
    train_logits = torch.from_numpy(train_logits)
    train_labels = torch.from_numpy(train_labels)
    print('train_logits.size()', train_logits.size())
    print('train_labels.size()', train_labels.size())
    print('train_logits[0], train_labels[0]: ', train_logits[0], train_labels[0])
    
    sigmoid = torch.nn.Sigmoid()
    train_labels = train_labels.long()
    train_logits = sigmoid(train_logits).float()
    train_probs = torch.abs((1 - train_labels) - train_logits)
    print('train_labels', train_labels)
    print('train_probs', train_probs)
    
    ##############################################
    #               Get sorted conf              #
    ##############################################
    if sel_type == 'topr':
        # select top r% by each class
        path = 'param/selection_id.npy'
        if not osp.exists(path):
            print('Computing ranking...It might take a while')
            selection_id = []
            for af in torch.unique(train_labels):
                per_cls_rank = torch.argsort(-train_probs[train_labels == af])
                per_cls_rank = per_cls_rank[:int(len(per_cls_rank) * args.threshold)]
                selection_id.append(torch.arange(len(train_probs))[train_labels == af][per_cls_rank])
            selection_id = torch.concat(selection_id).numpy()
            torch.save(selection_id, path)
        print('Done!')
        selection_id = torch.load(path)
    elif sel_type == 'value':
        selection = train_logits > args.threshold
        selection_id = np.arange(len(selection))[selection]
    else:
        raise NotImplemented

    selected_df = train_ds.iloc[selection_id, :]
    selected_df = selected_df.reset_index(drop=True)
    print('train_ds', train_ds, len(train_ds))
    print('selected_df', selected_df, len(selected_df))
    
    train_loader = get_dataloader(data=selected_df, cv_splits = None, 
                                  peptide_repr = args.peptide_repr, 
                                  mhc_repr = args.mhc_repr,
                                  batch_size = args.batch_size,
                                  shuffle = True, return_df=False)
    
    return train_loader



    
    # with torch.no_grad():
    #     train_logits = []
    #     train_labels = []
    #     for data in unshuffled_train_loader:
    #         peptide = data['peptide'].long().to(device)
    #         mhc = data['mhc'].long().to(device)
    #         affinity = data['BA'].long().to(device)
    #         print(peptide.size(), mhc.size(), affinity.size())
    #         pred_affinity = sigmoid(model(peptide, mhc))
    #         prob = torch.zeros(pred_affinity.size(0), 2).to(device)
    #         prob[:, 1] = pred_affinity
    #         prob[:, 0] = 1 - pred_affinity
    #         train_logits.append(prob[torch.arange(len(prob)), affinity].view(-1).long())
    #         train_labels.append(affinity)
        
    #     train_logits = torch.concat(train_logits).float()
    #     train_labels = torch.concat(train_labels).long()
    