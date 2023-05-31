"""
Code to generate the MA classifciation data
by separating the allele list into separate rows. 
Saves as new dataframe.
8million rows x 6 items in list = massive dataframe.
"""

import pandas as pd 
import ipdb 
from tqdm import tqdm 

if __name__=='__main__':
    df = pd.read_csv('./data/IEDB_classification_data.csv')
    df_ = df[df.n_possible_alleles!=1]
    
    
    # Positive presenting / samples 
    df = df_[df_.presented==1]
    df['mhc_pseudo_seq'] = df['mhc_pseudo_seq'].str.split(',')
    df['allele'] = df['allele'].str.split(',')
    dict = {
        'mhc_pseudo_seq': [allele for alleles in df['mhc_pseudo_seq'] for allele in alleles],  # Flatten the alleles lists
        'allele': [allele for alleles in df['allele'] for allele in alleles]  # Flatten the alleles lists
    }
    for c in df.columns:
        if c != 'mhc_pseudo_seq' and c!='allele':   
            dict[c] = df[c].repeat(df['mhc_pseudo_seq'].str.len())
    pos_df = pd.DataFrame(dict).reset_index(drop=True)


    # Repeat for negative presenting / samples 
    df = df_[df_.presented==0]
    df['mhc_pseudo_seq'] = df['mhc_pseudo_seq'].str.split(',')
    df['allele'] = df['allele'].str.split(',')
    dict = {
        'mhc_pseudo_seq': [allele for alleles in df['mhc_pseudo_seq'] for allele in alleles] , # Flatten the alleles lists
            'allele': [allele for alleles in df['allele'] for allele in alleles]  # Flatten the alleles lists

    }
    for c in df.columns:
        if c != 'mhc_pseudo_seq' and c!='allele':   
            dict[c] = df[c].repeat(df['mhc_pseudo_seq'].str.len())
    neg_df = pd.DataFrame(dict).reset_index(drop=True)


    df = pd.concat([neg_df, pos_df])
    df = df.rename(columns={'presented': 'affinity', 'cell_line': 'mhc_name'})
    df.to_csv('./data/MA_data.csv', index=False)
