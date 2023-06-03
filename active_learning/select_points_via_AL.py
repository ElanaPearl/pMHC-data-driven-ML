
import pandas as pd 
import numpy as np 
import ipdb

class SelectPointsAL:
    def __init__(self,df, n=1000, pos_perc=.056):
        self.df = df 
        self.n_pos = int(n*pos_perc)
        self.n_neg = n - self.n_pos
        self.n = n 

    def get(self):
        pass 
    
    def save(self, old_points, new_points, path):
        df = pd.concat([old_points, new_points])
        print(df.affinity.mean())
        df.to_csv(path, index=False)

class Random_SelectPoints(SelectPointsAL):

    def get(self):
        return self.df.sample(n=self.n) 
class RandomPosOnly_SelectPoints(SelectPointsAL):

    def get(self):
        df = self.df[self.df.affinity==1]
        return df.sample(n=self.n) 


class MostCertain_SelectPoints(SelectPointsAL):

    def get(self):
        # Entropy - higher = most certain 
        df = self.df[self.df.pred_affinity>.426]
        score = df.pred_affinity
        entropy = -score * (1-score)
        idx = np.argsort(entropy) # higher at bottom 
        most_certain_idx = idx[-self.n_pos:]
        pos_df = df.iloc[most_certain_idx]

        if self.n_neg > 0:
            df = self.df[self.df.pred_affinity<.426]
            score = df.pred_affinity
            entropy = -score * (1-score)
            idx = np.argsort(entropy) # higher at bottom 
            most_certain_idx = idx[-self.n_neg:]
            neg_df = df.iloc[most_certain_idx]
            return pd.concat([pos_df, neg_df])
        else:
            return pos_df   

class LeastCertain_SelectPoints(SelectPointsAL):

    def get(self):
        # Entropy - higher = most certain 
        df = self.df[self.df.pred_affinity>.426]
        score = df.pred_affinity
        entropy = -score * (1-score)
        idx = np.argsort(entropy) # higher at bottom 
        most_certain_idx = idx[:self.n_pos]
        pos_df = df.iloc[most_certain_idx]
        if self.n_neg > 0:
            df = self.df[self.df.pred_affinity<.426]
            score = df.pred_affinity
            entropy = -score * (1-score)
            idx = np.argsort(entropy) # higher at bottom 
            most_certain_idx = idx[:self.n_neg]
            neg_df = df.iloc[most_certain_idx]
            return pd.concat([pos_df, neg_df])
        else:
            return pos_df   

for n in ['10k', '100k', 'All']:
    print()
    print(n)
    tr_df = pd.read_csv(f'./active_learning/data/AL_n{n}_v0.csv')
    print(tr_df.shape)
    tr_df['data_source'] = 'SA split 0'
    pos_perc = tr_df.affinity.mean()
    embed_path = f'AL_n{n}_SAMAv1'
    df = pd.read_csv(f'./embeds/{embed_path}.csv')
    df['cv_split'] = 1
    df['data_source'] = 'AL points, SAMAv1 split 1'
    df = df[df.columns[1:]]

    fetcher = MostCertain_SelectPoints(df, pos_perc=pos_perc)
    AL_points = fetcher.get()
    path = f'./active_learning/data/AL_n{n}_v1_topCertain_balanced.csv'
    fetcher.save(tr_df, AL_points, path)

    fetcher = MostCertain_SelectPoints(df, pos_perc=1)
    AL_points = fetcher.get()
    path = f'./active_learning/data/AL_n{n}_v1_topCertain_posOnly.csv'
    fetcher.save(tr_df, AL_points, path)

    fetcher = Random_SelectPoints(df)
    AL_points = fetcher.get()
    path = f'./active_learning/data/AL_n{n}_v1_random.csv'
    fetcher.save(tr_df, AL_points, path)

    fetcher = RandomPosOnly_SelectPoints(df)
    AL_points = fetcher.get()
    path = f'./active_learning/data/AL_n{n}_v1_randomPosOnly.csv'
    fetcher.save(tr_df, AL_points, path)


    fetcher = LeastCertain_SelectPoints(df, pos_perc=pos_perc)
    AL_points = fetcher.get()
    path = f'./active_learning/data/AL_n{n}_v1_leastCertain_balanced.csv'
    fetcher.save(tr_df, AL_points, path)

    fetcher = LeastCertain_SelectPoints(df, pos_perc=1)
    AL_points = fetcher.get()
    path = f'./active_learning/data/AL_n{n}_v1_leastCertain_posOnly.csv'
    fetcher.save(tr_df, AL_points, path)


