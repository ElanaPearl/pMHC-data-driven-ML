"""
Merges multiple allele data with single allele data.
Note MA data was generated separately by Elana (and downloaded from google drive)
using predictions from external model 

Merges MA + SA and outputs to new df. 
"""

import pandas as pd 
import numpy as np 

if __name__ == "__main__":
    np.random.seed(42)
    sa = pd.read_csv('./data/IEDB_classification_data_SA.csv')

    for v in ['1','2']:
        ma = pd.read_csv(f'./data/IEDB_classification_MA_v{v}.csv')
        ma = ma.rename(columns={'cell_line': 'mhc_name'}).drop(columns='confidence')
        ma['cv_split'] = np.random.randint(5, size=len(ma))
        v_data = pd.concat([sa, ma])
        v_data.to_csv(f'./data/IEDB_classification_SA_MA_v{v}_highConf.csv')
        print(v_data.affinity.mean())
