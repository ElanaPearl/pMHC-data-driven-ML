import pandas as pd 
import ipdb 
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np 
from vocab import *


def load_blosum_as_dict(bl_type=62):
    """ 
    Load blosum matrix as dictionary.
    """
    import blosum # install package if you want to use blosum featurization  

    bl = blosum.BLOSUM(bl_type)
    dict_aa_repr = {}
    for i, aai in enumerate(AA_LIST):
        dict_aa_repr[aai] = []
        for j, aaj in enumerate(AA_LIST):
            dict_aa_repr[aai].append(bl[aai][aaj])
    return dict_aa_repr


class pMHCDataset(Dataset):
    def __init__(self, df_path: str = None, data: Any = None, cv_splits: Any = None,
                    peptide_repr: str = '1hot',
                    mhc_repr: str = '1hot',
                    max_peptide_len: int = 15,
                    sample: int = 0):
        """
        Custom Dataset class for loading pMHC dataset.

        Args:
            df_path (str): string path to load dataframe 
            cv_splits (Any): What subset of data to use based on the cv_split.
                        If cv_splits == None (default), then use entire dataframe.
                        If cv_splits == int (ie. 4), then only use data where cv_split=4.
                        If cv_splits == List[int], then use data where cv_split.isin(List[int]).
            peptide_repr (str): how to represent the peptide AA data 
            mhc_repr (str): how to represent the MHC AA data 
            max_peptide_len (int): maximum peptide length, for padding
                string - no changes, just leave sequence as stirng
                1hot - create 20-dim 1hot vector for each AA in the sequence
                blosum - vectorize according to blosum matrix
                indices - like 1hot, but just put in index into a dictionary instead of 1hot
        """


        # Data = all data in the listed CV_splits; by default
        # includes all data
        assert not (df_path is None and data is None)
        if df_path is not None:
            self.data = pd.read_csv(df_path)
        else:
            self.data = data
        if 'mhc_pseudo_seq' not in self.data.columns:
            self.data = self.data.rename(columns={'mhc': 'mhc_pseudo_seq'})
            self.data.to_csv(df_path)
        if type(cv_splits) == int:
            cv_splits = [cv_splits]
        elif cv_splits is None:
            cv_splits = self.data.cv_split.unique()
        # import ipdb; ipdb.set_trace()
        self.data = self.data[self.data.cv_split.isin(cv_splits)]
        if sample > 0:
            self.data = self.data.sample(n=sample)
        # import ipdb; ipdb.set_trace() 
        # self.data.to_csv('./active_learning/data/AL_n20k_v0.csv',index=False)

        # Set up for optional preprocessing / representing peptide and MHC AA data 
        self.peptide_repr = peptide_repr
        self.max_peptide_len = max_peptide_len
        self.mhc_repr = mhc_repr
        allowable_repr = ['1hot', 'blosum62', 'string', 'indices']
        assert self.peptide_repr in allowable_repr, \
                    f"peptide_repr must be in options {allowable_repr}, found {peptide_repr}"
        if self.peptide_repr == '1hot' or self.mhc_repr == '1hot':
            self.aa_encoder = OneHotEncoder(sparse_output=False).fit(AA_LIST.reshape(-1,1))
        if self.peptide_repr == 'blosum62' or self.mhc_repr == 'blosum62':
            self.bl_dict = load_blosum_as_dict(bl_type=62)
        if self.peptide_repr == 'indices' or self.mhc_repr == 'indices':
            self.aa_encoder = LabelEncoder().fit(AA_LIST.reshape(-1,1))

    def _get_aa_1hot_repr(self, aa_sequence: str, repr: str, pad_to = None): 
        """
        Get string AA representaiton and convert it to 1hot repr
        """
        aa_list = np.array(list(aa_sequence)).reshape(-1,1)
        output = self.aa_encoder.transform(aa_list) 
        seq_len_b4_pad = output.shape[0]
        if pad_to is not None and pad_to - output.shape[0] > 0:
            n = pad_to - output.shape[0]
            output = np.concatenate([output, np.ones((n, output.shape[1])) * PAD_ID])
        return output, seq_len_b4_pad

    def _get_aa_indices_repr(self, aa_sequence: str, repr: str, pad_to = None): 
        """
        Get string AA representaiton and convert it to categorical labels / indices
        matched with where the AA appears in AA_LIST
        """
        aa_list = np.array(list(aa_sequence)).reshape(-1,1)
        aa_list = aa_list.squeeze().ravel()
        output = self.aa_encoder.transform(aa_list) 
        seq_len_b4_pad = output.shape[0]
        if pad_to is not None and pad_to - output.shape[0] > 0:
            n = pad_to - output.shape[0]
            output = np.concatenate([output, np.ones((n,)) * PAD_ID])
        return output, seq_len_b4_pad

    def _get_blosum_repr(self, aa_sequence: str):
        bl_repr = np.zeros((len(aa_sequence), 20)) 
        for i, aa in enumerate(aa_sequence): 
            bl_repr[i] = self.bl_dict[aa]
        return bl_repr 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        series = self.data.iloc[index]

        peptide = series.peptide.upper()
        mhc = series.mhc_pseudo_seq
        if 'U' in peptide:
            peptide = peptide.replace('U','X')
        if 'U' in mhc:
            mhc = mhc.replace('U', 'X')
            
        # Peptide representation 
        pep_len = None
        if self.peptide_repr =='1hot':
            peptide, pep_len = self._get_aa_1hot_repr(peptide,repr=self.peptide_repr, 
                                                      pad_to=self.max_peptide_len)
        elif self.peptide_repr == 'indices':
            peptide, pep_len = self._get_aa_indices_repr(peptide,repr=self.peptide_repr, 
                                            pad_to=self.max_peptide_len)
        elif self.peptide_repr == 'blosum62':
            peptide = self._get_blosum_repr(peptide)

        # MHC representatin 
        if self.mhc_repr =='1hot':
            mhc, _ = self._get_aa_1hot_repr(mhc,repr=self.mhc_repr)
        elif self.mhc_repr == 'indices':
            mhc, _ = self._get_aa_indices_repr(mhc,repr=self.mhc_repr)
        elif self.mhc_repr == 'blosum62':
            mhc = self._get_blosum_repr(mhc)

        return {'mhc_name': series.mhc_name, # MHC allele name
                'BA': series.affinity, #binding affinity 
                'peptide': peptide, #peptide representation 
                'mhc': mhc, #MHC representation
                'peptide_len': pep_len} # Length of original peptide (8-15) before 0-padding



def get_dataloader(df_path: str= None, 
                   data: Any = None,
                   cv_splits: Any = None,
                   peptide_repr: str = '1hot',
                   mhc_repr: str= '1hot',
                   batch_size: int = 32,
                   shuffle: bool = True,
                   return_df: bool = False,
                   sample: int = 0):
    """
    Get training / validation dataloaders using pMHCDataset class 
    """

    ds = pMHCDataset(df_path=df_path, data=data,
                     cv_splits=cv_splits, 
                     peptide_repr=peptide_repr, mhc_repr=mhc_repr, 
                     sample=sample)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    if return_df:
        return ds_loader, ds.data
    return ds_loader


if __name__ == '__main__': 
    ds = pMHCDataset(df_path='./data/IEDB_classification_data_SA.csv', cv_splits=None, 
                peptide_repr='indices', mhc_repr='indices', random_pad=True)
    ds_loader = get_dataloader(df_path='./data/IEDB_classification_data_SA.csv', cv_splits=None,
        peptide_repr='indices', mhc_repr='indices')
        
