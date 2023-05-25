import wandb
import argparse
from model_lib.initialise_model import initialise_model
from dataset import get_dataloader
import torch.optim as optim 
import ipdb 
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np 
import torch.nn as nn
import torch
from tqdm import tqdm
from vocab import * 
import ipdb 
import os 



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

    model_output = model_output.detach().cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()
    BA_true_mean = true_labels.mean()
    BA_pred_std = model_output.std()

    # true_labels = (true_labels > .5).astype(float)
    # model_output = np.clip(model_output, 0,1)
    metrics = {f"{folder}/loss": loss.item(), f'{folder}/BA_mean': BA_true_mean, f'{folder}/BA_std': BA_pred_std}

    if len(np.unique(true_labels)) == 2: 
        # only compute AUROC / AUPRC if postive class is present
        auroc = roc_auc_score(true_labels, model_output)
        auprc = average_precision_score(true_labels, model_output)
        metrics.update({f'{folder}/aucroc': auroc, f'{folder}/aucprc': auprc})
    
    wandb.log(metrics)

def train_pMHC(args):
    """
    Trains pMHC model.
    """

    # Create model, based on DeepVHPPI (bert)
    device =  "cuda" if args.use_cuda else 'cpu'
    if device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

    model = initialise_model(args, vocab_size=21+1, num_classes=1, device=device)
    model = model.to(device)
    # Loss fn = weighted BCE 
    weight = torch.tensor([args.pos_weight]).to(device)  # Higher weight for positive (minority) class = ~ 100 / 5 since 5% data is + 
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Load data - training data is the MA classifcation data. We ignore the 8ish million
    # multiple allele data and keep the 4ish million single allele data

    # train_loader = get_dataloader(args.tr_df_path, cv_splits = None, 
    #                               peptide_repr = args.peptide_repr, 
    #                               mhc_repr = args.mhc_repr,
    #                               batch_size = args.batch_size,
    #                               shuffle = True)
    
    # # Val data is split 4 of regression data. Splits 0,1,2,3 are for heldout testing. 
    # val_loader = get_dataloader(args.val_df_path, cv_splits = 4, 
    #                             peptide_repr = args.peptide_repr, 
    #                             mhc_repr = args.mhc_repr,
    #                             batch_size = args.batch_size,
    #                             shuffle = False)
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
            torch.save(model.state_dict(), f'{args.save_path}/{wandb.run.name}_ckpt_e{epoch}_i{i}.pth')
            print(2/0)
            model.train()
            peptide = data['peptide'].long().to(device)
            mhc = data['mhc'].long().to(device)
            affinity = data['BA'].float().to(device)
            if args.train_regression_thresh:
                affinity = (affinity > .426).float()
            pred_affinity = model(peptide, mhc)
            loss = loss_fn(pred_affinity, affinity)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
            if is_nan or torch.isnan(pred_affinity).any():
                continue
                import ipdb; ipdb.set_trace()
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
                        if args.train_regression_thresh or 'regression' in args.val_df_path:
                            affinity = (affinity > .426).float()
                        pred_affinity = model(peptide, mhc)
                        loss += loss_fn(pred_affinity, affinity)
                        affinity_lst.append(affinity.detach().cpu().view(-1))
                        pred_affinity_lst.append(pred_affinity.detach().cpu().view(-1))
                        
                average_loss = loss/len(val_loader)
                affinity_lst = torch.concat(affinity_lst)
                pred_affinity_lst = torch.concat(pred_affinity_lst)
                log_wandb(pred_affinity_lst, affinity_lst.long(), average_loss, folder='val')    
        
        torch.save(model.state_dict(), f'{args.save_path}/{wandb.run.name}_ckpt_e{epoch}_i{i}.pth')


if __name__ == '__main__':

    # Create the argument parser
    parser = argparse.ArgumentParser(description='pMHC Training Script')

    # Misc arguments
    parser.add_argument('-learning_rate', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('-n_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('-pos_weight', type=float, default=20., help='pos weight for BCE Loss')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-use_cuda', action='store_true', help='use cuda or cpu')
    parser.add_argument('-train_regression_thresh', action='store_true', help='train with regression data and threshold for classificaiton')
    parser.add_argument('-save_path', type=str, default='./ckpt/', help='Path to dump ckpts')

    # Data arguments
    parser.add_argument('-tr_df_path', type=str, default='./data/IEDB_classification_data_SA.csv', help='Path to load training dataframe')
    parser.add_argument('-val_df_path', type=str, default='./data/IEDB_regression_data.csv', help='Path to load val / test dataframe')
    parser.add_argument('-peptide_repr', type=str, default='indices', help='how to represent peptide, if at all') 
    parser.add_argument('-mhc_repr', type=str, default='indices', help='how to represent mhc allele, if at all') 

    # Model arguments
    parser.add_argument('-model', type=str, default='bert',choices=['mlp', 'bert', 'lstm'], help='type of model')
    parser.add_argument('-hidden', type=int, default=128, help='hidden size of transformer model')
    parser.add_argument('-layers', type=int, default=4, help='number of layers of bert')
    parser.add_argument('-attn_heads', type=int, default=4, help='number of attention heads in transformer')
    parser.add_argument('-seq_len', type=int, default=34, help='maximum sequence length') 
    parser.add_argument('-dropout', type=float, default=0.1, help='dropout rate') 
    parser.add_argument('-emb_type', type=str, default='lookup', 
                help='embedding type', choices=['lookup', 'conv', 'continuous', 'both', 'pair'])
    parser.add_argument('-activation', type=str, default='gelu', help='activation function') 

    # Parse the command-line arguments
    args = parser.parse_args()
    train_pMHC(args)
