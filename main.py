import wandb
import argparse

from dataset import get_dataloader
import torch.optim as optim 
import ipdb 
import numpy as np 
import torch.nn as nn
import torch
from tqdm import tqdm
from vocab import * 
import os 
import random 
from torch.utils.data import Dataset, DataLoader
from model_lib.initialise_model import initialise_model
from utils import log_wandb, load_pretrained, reweight_dataloader


def train_pMHC(args, device, train_loader=None):
    """
    Trains pMHC model.
    """
    
    model = initialise_model(args, vocab_size=VOCAB_SIZE, num_classes=1, device=device)
    model = model.to(device)
    # Loss fn = weighted BCE 
    weight = torch.tensor([args.pos_weight]).to(device)  # Higher weight for positive (minority) class = ~ 100 / 5 since 5% data is + 
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Load data - training data is the MA classifcation data. We ignore the 8ish million
    # multiple allele data and keep the 4ish million single allele data
    if train_loader == None:
        train_loader = get_dataloader(args.tr_df_path, cv_splits = None, 
                                    peptide_repr = args.peptide_repr, 
                                    mhc_repr = args.mhc_repr,
                                    batch_size = args.batch_size,
                                    shuffle = True)
    
    # Val data is split 4 of regression data. Splits 0,1,2,3 are for heldout testing. 
    val_loader = get_dataloader(args.val_df_path, cv_splits = 4, 
                                peptide_repr = args.peptide_repr, 
                                mhc_repr = args.mhc_repr,
                                batch_size = args.batch_size,
                                shuffle = False)
    wandb.init(
        # set the wandb project where this run will be logged
        project="pMHC",
        # track hyperparameters and run metadata
        config=args
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print(len(train_loader))
    for epoch in range(args.n_epochs):
        for i, data in enumerate(tqdm(train_loader)):

            model.train()
            peptide = data['peptide'].long().to(device)
            mhc = data['mhc'].long().to(device)
            affinity = data['BA'].float().to(device)
            pred_affinity = model(peptide, mhc)

            loss = loss_fn(pred_affinity, affinity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_wandb(pred_affinity, affinity, loss)
            
            if (i + 1) % 500 == 0:
                print(f"Epoch [{epoch + 1}/{args.n_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            if i % 1000 == 0:
                # Test the model on regression data 
                model.eval()
                with torch.no_grad():
                    affinity_lst = []
                    pred_affinity_lst = []
                    loss = 0
                    for data in val_loader:
                        peptide = data['peptide'].long().to(device)
                        mhc = data['mhc'].long().to(device)
                        affinity = data['BA'].float().to(device)
                        if 'regression' in args.val_df_path:
                            affinity = (affinity > .426).float()
                        pred_affinity = model(peptide, mhc)
                        loss += loss_fn(pred_affinity, affinity)
                        affinity_lst.append(affinity.detach().cpu().view(-1))
                        pred_affinity_lst.append(pred_affinity.detach().cpu().view(-1))
                        
                average_loss = loss/len(val_loader)
                affinity_lst = torch.concat(affinity_lst)
                pred_affinity_lst = torch.concat(pred_affinity_lst)
                log_wandb(pred_affinity_lst, affinity_lst.long(), average_loss, folder='val')    
        
        torch.save(model.state_dict(), f'{args.save_path}/{wandb.run.name}_ckpt_e{epoch}.pth')


if __name__ == '__main__':

    # Create the argument parser
    parser = argparse.ArgumentParser(description='pMHC Training Script')

    # Misc arguments
    parser.add_argument('-learning_rate', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('-n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('-pos_weight', type=float, default=20., help='pos weight for BCE Loss; should be 1/(% pos in data)')
    parser.add_argument('-batch_size', type=int, default=512, help='batch size')
    parser.add_argument('-seed', type=int, default=42, help='seed; oddly super important - other seeds not 42 do not work')
    parser.add_argument('-device', type=int, default=0, help='cuda device')
    parser.add_argument('-save_path', type=str, default='./ckpt/', help='Path to dump ckpts')

    # Data arguments 
    parser.add_argument('-tr_df_path', type=str, default='./data/IEDB_classification_data_SA.csv', help='Path to load training dataframe') #'./data/IEDB_classification_data_SA.csv'
    parser.add_argument('-val_df_path', type=str, default='./data/IEDB_regression_data.csv', help='Path to load val / test dataframe')
    parser.add_argument('-peptide_repr', type=str, default='indices', help='how to represent peptide, if at all') 
    parser.add_argument('-mhc_repr', type=str, default='indices', help='how to represent mhc allele, if at all') 

    # Model arguments
    parser.add_argument('-model', type=str, default='lstm',choices=['mlp', 'bert', 'lstm'], help='type of model')
    parser.add_argument('-hidden', type=int, default=64, help='hidden size of transformer model')
    parser.add_argument('-embed_dim', type=int, default=100, help='hidden size of transformer model')
    parser.add_argument('-layers', type=int, default=3, help='number of layers of bert')
    parser.add_argument('-dropout', type=float, default=0.0, help='dropout rate') 
    parser.add_argument('-model_path', type=str, default=None, help='pretrained model path')
    parser.add_argument('-threshold', type=float, default=0.9, help='threshold for selection') 

    # Parse the command-line arguments
    args = parser.parse_args()
    
    device =  f"cuda:{args.device}"
    print(f'Using {device} for training')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wandb.init(
        # set the wandb project where this run will be logged
        project="pMHC",
        # track hyperparameters and run metadata
        config=args
    )
    if args.model_path is not None:
        model = load_pretrained(args, args.model_path, device)
        train_loader = reweight_dataloader(args, model, device)
    else:
        train_loader = None
    train_pMHC(args, device, train_loader)