# import wandb
import argparse
from model_lib.initialise_model import initialise_model
from dataset import get_dataloader
import torch.optim as optim 
import ipdb 
from sklearn.metrics import roc_auc_score, average_precision_score

def log_wandb(model_output, true_labels, loss, tr_val_test='train'):
    model_output = model_output.detach().cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()
    auroc = roc_auc_score(true_labels, model_output)
    auprc = average_precision_score(true_labels, model_output)
    metrics = {f"{tr_val_test}/loss": loss.item(), 
                f'{tr_val_test}/aucroc': auroc, f'{tr_val_test}/aucprc': auprc}
    
    wandb.log(metrics)

def train_pMHC(args):
    device =  "cuda:0" if args.use_cuda else 'cpu'
    model, loss_fn = initialise_model(args, vocab_size=20, num_classes=1, device=device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_loader = get_dataloader(args.tr_df_path, cv_splits = None, 
                        peptide_repr = args.peptide_repr, mhc_repr = args.mhc_repr,
                        batch_size = args.batch_size)
    
    val_loader = get_dataloader(args.val_df_path, cv_splits = 4, 
                        peptide_repr = args.peptide_repr, mhc_repr = args.mhc_repr,
                        batch_size = args.batch_size)

    for epoch in range(args.n_epochs):
        for i, data in enumerate(train_loader):
            peptide = data['peptide'].to(device)
            mhc = data['mhc'].to(device)
            affinity = data['BA']

            pred_affinity = model(peptide, mhc)
            loss = loss_fn(pred_affinity, affinity)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            log_wandb(pred_affinity, affinity, loss)
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in val_loader:
            peptide = data['peptide'].to(device)
            mhc = data['mhc'].to(device)
            affinity = data['BA']

            pred_affinity = model(peptide, mhc)
            loss = loss_fn(pred_affinity, affinity)
            
            
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
    model.train()

if __name__ == '__main__':

    # Create the argument parser
    parser = argparse.ArgumentParser(description='pMHC Training Script')

    # Misc arguments
    parser.add_argument('-learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('-n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('-batch_size', type=int, default=100, help='batch size')
    parser.add_argument('-use_cuda', action='store_true', help='use cuda or cpu')

    # Data arguments
    parser.add_argument('-tr_df_path', type=str, default='./data/IEDB_classification_data_SA.csv', help='Path to load training dataframe')
    parser.add_argument('-val_df_path', type=str, default='./data/IEDB_regression_data.csv', help='Path to load val / test dataframe')
    parser.add_argument('-peptide_repr', type=str, default='string', help='how to represent peptide, if at all') 
    parser.add_argument('-mhc_repr', type=str, default='string', help='how to represent mhc allele, if at all') 

    # Model arguments
    parser.add_argument('-hidden', type=int, default=256, help='hidden size of transformer model')
    parser.add_argument('-layers', type=int, default=6, help='number of layers of bert')
    parser.add_argument('-attn_heads', type=int, default=4, help='number of attention heads in transformer')
    parser.add_argument('-seq_len', type=int, default=20, help='maximum sequence length') # TODO 
    parser.add_argument('-dropout', type=float, default=.1, help='dropout rate') 
    parser.add_argument('-emb_type', type=str, default='conv', 
                help='embedding type', choices=['lookup', 'conv', 'continuous', 'both', 'pair']) # TODO 
    parser.add_argument('-activation', type=str, default='gelu', help='activation function') 

    # Parse the command-line arguments
    args = parser.parse_args()
    train_pMHC(args)
