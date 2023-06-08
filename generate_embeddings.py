import torch
import torch.nn as nn
from model_lib.initialise_model import initialise_model
from dataset import get_dataloader
from vocab import *
import argparse 
import os 
from tqdm import tqdm 
import random 
from sklearn.metrics import roc_auc_score, average_precision_score
import torch 

def generate_embeds(args):
    device =  "cuda" if args.use_cuda else 'cpu'
    # if device == 'cuda':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    model = initialise_model(args, vocab_size=VOCAB_SIZE, num_classes=1, device=device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    model = model.to(device)

    loader, df = get_dataloader(args.df_path, cv_splits = args.cv_split, 
                                  peptide_repr = args.peptide_repr, 
                                  mhc_repr = args.mhc_repr,
                                  batch_size = args.batch_size,
                                  shuffle = False,
                                  return_df = True)

    if not os.path.exists('./embeds/'):
        os.makedirs('./embeds/')
    if args.save_path == '':
        file = args.model_path.split('/')[-1].split('.pth')[0]
        if 'regress' in args.df_path:
            file += '_rgrssnData'
        elif 'classification' in args.df_path:
            file += '_clssfctnData'
        elif 'MA_data' in args.df_path:
            file += '_clssfctnMAData'
        else:
            print("Unknown data type, neither class or regress dataset!")
        args.save_path = f'./embeds/{file}.csv'
        
    latents = [] # latents, penultimate layer of model
    pred_affinities = []
    with torch.no_grad():
        for data in tqdm(loader): 
            peptide = data['peptide'].long().to(device)
            mhc = data['mhc'].long().to(device)
            affinity = data['BA'].float().to(device)
            
            pred_affinity, z = model(peptide, mhc, return_z=True)
            pred_affinities.append(pred_affinity.detach().cpu().numpy())
            latents.append(z.detach().cpu().numpy())
           
    pred_affinities = np.concatenate(pred_affinities)
    latents = np.concatenate(latents,axis=0)

    df['pred_affinity_logits'] = pred_affinities
    for i in range(latents.shape[1]):
        column_name = f"latent_{i}"
        df[column_name] = latents[:,i]

    sigmoid = torch.nn.Sigmoid()
    pred = sigmoid(torch.Tensor(pred_affinities)).numpy()
    df['pred_affinity'] = pred
    label = (df.affinity > .426).astype(int)
    print(roc_auc_score(label, pred),average_precision_score(label, pred))

    # import ipdb; ipdb.set_trace()
    if args.cv_split is not None:
         args.save_path = args.save_path.split('.csv')[0] + f'_{args.cv_split}.csv'
    # df.to_csv(path, index=False)
    df.to_csv(args.save_path, index=False)

if __name__ == '__main__':

    # Create the argument parser
    parser = argparse.ArgumentParser(description='pMHC Training Script')

    # Misc arguments 
    parser.add_argument('-model_path', type=str, default='./ckpt/sage-haze-125_ckpt_e37_i16067.pth')
    parser.add_argument('-batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-seed', type=int, default=42, help='seed')
    parser.add_argument('-cv_split', default=None, help='cv_split if any to consider, useful for big datasets')
    parser.add_argument('-use_cuda', action='store_true', help='use cuda or cpu')
    parser.add_argument('-save_path', type=str, default='', help='Path to dump ckpts')
    parser.add_argument('-cuda_devices', type=str, default='0')

    # Data arguments 
    parser.add_argument('-df_path', type=str, default='./data/IEDB_classification_SA_MA_v1.csv', help='Path to load training dataframe')
    parser.add_argument('-peptide_repr', type=str, default='indices', help='how to represent peptide, if at all') 
    parser.add_argument('-mhc_repr', type=str, default='indices', help='how to represent mhc allele, if at all') 

    # Model arguments 
    parser.add_argument('-hidden', type=int, default=64, help='hidden size of transformer model')
    parser.add_argument('-embed_dim', type=int, default=100, help='hidden size of transformer model')
    parser.add_argument('-model', type=str, default='lstm',choices=['mlp', 'bert', 'lstm'], help='type of model')
    parser.add_argument('-layers', type=int, default=3, help='number of layers of bert')
    # Parse the command-line arguments
    args = parser.parse_args()
    args.dropout = 0
    if args.cv_split is not None:
        args.cv_split = int(args.cv_split)
    generate_embeds(args)

