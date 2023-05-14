import pandas as pd 
import ipdb 
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any 
from sklearn.preprocessing import OneHotEncoder
import numpy as np 


class pMHCDataset(Dataset):
    def __init__(self, df_path: str, cv_splits: Any = None,
                    peptide_repr: str = '1hot',
                    mhc_repr: str = '1hot'):
        """
        Custom Dataset class for loading pMHC dataset.

        Args:
            df_path (str): string path to load dataframe 
            cv_splits (Any): What subset of data to use based on the cv_split.
                        If cv_splits == None (default), then use entire dataframe.
                        If cv_splits == int (ie. 4), then only use data where cv_split=4.
                        If cv_splits == List[int], then use data where cv_split.isin(List[int]).
            peptide_repr (str): how to represent the peptide data 
            mhc_repr (str): how to represent the peptide data 
        """

        self.data = pd.read_csv(df_path)

        if type(cv_splits) == int:
            cv_splits = [cv_splits]
        elif cv_splits is None:
            cv_splits = self.data.cv_split.unique()
        
        self.data = self.data[self.data.cv_split.isin(cv_splits)]

        # Set up for preprocessing / representing peptide and MHC data 
        self.peptide_repr = peptide_repr
        self.mhc_repr = mhc_repr
        assert self.peptide_repr in ['1hot'], f"peptide_repr must be in options ['1hot'], found {peptide_repr}"
        if self.peptide_repr == '1hot' or self.mhc_repr == '1hot':
            AAs = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
            self.aa_encoder = OneHotEncoder(sparse_output=False).fit(AAs.reshape(-1,1))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        series = self.data.iloc[index]
        
        peptide = series.peptide
        mhc = series.mhc_psuedo_seq
        if self.peptide_repr == '1hot':
            pep_list = np.array(list(peptide)).reshape(-1,1)
            peptide = self.aa_encoder.transform(pep_list)

        if self.mhc_repr == '1hot':
            mhc_list = np.array(list(mhc)).reshape(-1,1)
            mhc = self.aa_encoder.transform(mhc_list)

        return {'mhc_name': series.mhc_name,
                'BA': series.affinity,
                'peptide': peptide,
                'mhc': mhc}



def get_dataloaders(df_path: str, cv_splits_tr: Any = None,
                    cv_splits_val: Any = None,
                    peptide_repr: str = '1hot',
                    batch_size: int = 32):
    """
    Get training / validation dataloaders using pMHCDataset class 
    """
    tr_ds = pMHCDataset(df_path, cv_splits_tr, peptide_repr)
    val_ds = pMHCDataset(df_path, cv_splits_val, peptide_repr)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return tr_loader, val_loader



if __name__ == '__main__': 
    ds = pMHCDataset(df_path='./data/IEDB_regression_data.csv', cv_splits=None)
    ds[0]
