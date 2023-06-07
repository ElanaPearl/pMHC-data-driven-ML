import os 
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
import pandas as pd 

def generate_embeds(args):
    device =  "cuda" #if args.use_cuda else 'cpu'
    # if device == 'cuda':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    loader, df = get_dataloader(args.df_path, cv_splits = [0,1,2,3], 
                            peptide_repr = args.peptide_repr, 
                            mhc_repr = args.mhc_repr,
                            batch_size = args.batch_size,
                            shuffle = False,
                            return_df = True)

    directory = './ckpt/'
    file_list = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)) and 'n5k_' in file:
            file = file.split('_ckpt_e')[0]
            file_list.append(file)
    file_list = set(file_list)
    import ipdb; ipdb.set_trace()
    try:
        results_df = pd.read_csv('results_df_5k.csv')
    except Exception as e:
        print(e)
        results_df = pd.DataFrame({'model': [], 'aucroc': [], 'aucprc': [], 'base_n':[], 'n_added': [], 'how_added': []})

    for name in file_list:
        
        args.model_path = f"./ckpt/{name.split('.csv')[0]}_ckpt_e199.pth"
        if 'samples' in args.model_path:
            args.model_path = args.model_path.replace('samples', 'sample')
        if args.model_path in np.array(results_df.model):
            print(f"{args.model_path} already in df!")
            continue 
        print(args.model_path)
        try:
            model = initialise_model(args, vocab_size=VOCAB_SIZE, num_classes=1, device=device)
            model.load_state_dict(torch.load(args.model_path))
        except Exception as e:
            print(e, name)
            continue

        model.eval()
        model = model.to(device)

        latents = [] # latents, penultimate layer of model
        pred_affinities = []
        with torch.no_grad():
            for data in tqdm(loader): 
                peptide = data['peptide'].long().to(device)
                mhc = data['mhc'].long().to(device)
                affinity = data['BA'].float().to(device)
                
                pred_affinity, z = model(peptide, mhc, return_z=True)
                pred_affinities.append(pred_affinity.detach().cpu().numpy())
           
        pred_affinities = np.concatenate(pred_affinities)
        sigmoid = torch.nn.Sigmoid()
        pred = sigmoid(torch.Tensor(pred_affinities)).numpy()
        label = (df.affinity > .426).astype(int)
        roc, prc = roc_auc_score(label, pred),average_precision_score(label, pred)
        results = {'model': [args.model_path], 'aucroc': [roc], 'aucprc': [prc], 
                        'n_added': [name.split('sample')[1].split('_')[0]],
                        'how_added': [name.split('sample')[0].split('v1')[1]], 
                        'base_n': [name.split('v1')[0].split('_n')[1]]}
        results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)
        print(args.model_path, roc, prc)

    

    import ipdb; ipdb.set_trace()
    # results_df.to_csv('results_df.csv')
    # if args.cv_split is not None:
    #      args.save_path = args.save_path.split('.csv')[0] + f'_{args.cv_split}.csv'
    # df.to_csv(path, index=False)
    # df.to_csv(args.save_path, index=False)

if __name__ == '__main__':

    # Create the argument parser
    parser = argparse.ArgumentParser(description='pMHC Training Script')

    # Misc arguments 
    parser.add_argument('-batch_size', type=int, default=256, help='batch size')
    parser.add_argument('-seed', type=int, default=42, help='seed')
    parser.add_argument('-cv_split', default=None, help='cv_split if any to consider, useful for big datasets')
    parser.add_argument('-save_path', type=str, default='', help='Path to dump ckpts')
    parser.add_argument('-cuda_devices', type=str, default='0')

    # Data arguments 
    parser.add_argument('-df_path', type=str, default='./data/IEDB_regression_data.csv', help='Path to load training dataframe')
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



