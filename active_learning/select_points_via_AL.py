
import pandas as pd 
import numpy as np 
import ipdb
from analysis.methods import TrustScore 

THRESH = .5 

class SelectPointsAL:
    def __init__(self,df, n=1000, pos_perc=.056):
        self.df = df 
        self.n_pos = int(n*pos_perc)
        self.n_neg = n - self.n_pos
        self.n = n 

    def get(self):
        pass 
    
    def save(self, old_points, new_points, path):
        new_points = new_points[old_points.columns]
        df = pd.concat([old_points, new_points])
        df.index = np.arange(len(df))
        df['pos_perc'] = df.affinity.mean()
        print(df.shape)
        print(df.affinity.mean())
        df.to_csv(path, index=False)

class Random_SelectPoints(SelectPointsAL):

    def get(self):
        return self.df.sample(n=self.n) 

class RandomPosOnly_SelectPoints(SelectPointsAL):

    def get(self):
        df = self.df[self.df.pred_affinity>THRESH]
        return df.sample(n=self.n) 


class MostCertain_SelectPoints(SelectPointsAL):

    def get(self):
        # Entropy - higher = most certain 
        df = self.df[self.df.pred_affinity>.THRESH]
        score = df.pred_affinity
        entropy = -score * (1-score)
        idx = np.argsort(entropy) # higher at bottom 
        most_certain_idx = idx[-self.n_pos:]
        pos_df = df.iloc[most_certain_idx]

        if self.n_neg > 0:
            df = self.df[self.df.pred_affinity<.THRESH]
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
        df = self.df[self.df.pred_affinity>THRESH]
        score = df.pred_affinity
        entropy = -score * (1-score)
        idx = np.argsort(entropy) # higher at bottom 
        most_certain_idx = idx[:self.n_pos]
        pos_df = df.iloc[most_certain_idx]
        if self.n_neg > 0:
            df = self.df[self.df.pred_affinity<THRESH]
            score = df.pred_affinity
            entropy = -score * (1-score)
            idx = np.argsort(entropy) # higher at bottom 
            most_certain_idx = idx[:self.n_neg]
            neg_df = df.iloc[most_certain_idx]
            return pd.concat([pos_df, neg_df])
        else:
            return pos_df   

class TrustScore_MostCertain_SelectPoints(SelectPointsAL):
    def set_pos_perc(self, n, pos_perc):
        self.n_pos = int(n*pos_perc)
        self.n_neg = n - self.n_pos
        self.n = n 

    def __init__(self,df, n=1000, pos_perc=.056):
        self.df = df 

        self.set_pos_perc(n, pos_perc)

        tscore = TrustScore()
        embed_cols = [c for c in self.df.columns if 'latent_' in c]
        embeds = self.df[embed_cols].to_numpy()
        preds = np.array((self.df.pred_affinity > THRESH).astype(int))
        labels = np.array((self.df.affinity > THRESH).astype(int))
        tscore.fit(embeddings=embeds, predictions=preds, labels=labels)
        self.scores = tscore.get_score(embeddings=embeds, predictions=preds, labels=labels)

    def get(self):
        
        
        df = self.df[self.df.pred_affinity>THRESH]
        score = self.scores[self.df.pred_affinity>THRESH]
        idx = np.argsort(score) # higher at bottom 
        most_certain_idx = idx[-self.n_pos:] 
        pos_df = df.iloc[most_certain_idx]
        
        if self.n_neg > 0:
            df = self.df[self.df.pred_affinity<THRESH]
            score = self.scores[self.df.pred_affinity<THRESH]
            idx = np.argsort(score) # higher at bottom 
            least_certain_idx = idx[-self.n_neg:]
            neg_df = df.iloc[least_certain_idx]
            return pd.concat([pos_df, neg_df])
        else:
            return pos_df  

class TrustScore_LeastCertain_SelectPoints(SelectPointsAL):
    def set_pos_perc(self, n, pos_perc):
        self.n_pos = int(n*pos_perc)
        self.n_neg = n - self.n_pos
        self.n = n 

    def __init__(self,df, n=1000, pos_perc=.056):
        self.df = df 

        self.set_pos_perc(n, pos_perc)

        tscore = TrustScore()
        embed_cols = [c for c in self.df.columns if 'latent_' in c]
        embeds = self.df[embed_cols].to_numpy()
        preds = np.array((self.df.pred_affinity > THRESH).astype(int))
        labels = np.array((self.df.affinity > THRESH).astype(int))
        tscore.fit(embeddings=embeds, predictions=preds, labels=labels)
        self.scores = tscore.get_score(embeddings=embeds, predictions=preds, labels=labels)

    def get(self):
        
        
        df = self.df[self.df.pred_affinity>THRESH]
        score = self.scores[self.df.pred_affinity>THRESH]
        idx = np.argsort(score) # higher at bottom 
        most_certain_idx = idx[:self.n_pos] 
        pos_df = df.iloc[most_certain_idx]
        
        if self.n_neg > 0:
            df = self.df[self.df.pred_affinity<THRESH]
            score = self.scores[self.df.pred_affinity<THRESH]
            idx = np.argsort(score) # higher at bottom 
            least_certain_idx = idx[:self.n_neg]
            neg_df = df.iloc[least_certain_idx]
            return pd.concat([pos_df, neg_df])
        else:
            return pos_df  

if __name__ == '__main__': #export PYTHONPATH="${PYTHONPATH}:/lfs/turing4/local/karaliu/pMHC-data-driven-ML/"
    for n in ['100k', 'All', '10k', '100k']:
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

        # fetcher = TrustScore_SelectPoints(df, pos_perc=pos_perc)
        # AL_points = fetcher.get()
        # path = f'./active_learning/data/AL_n{n}_v1_trustScore_balanced.csv'
        # fetcher.save(tr_df, AL_points, path)

        # fetcher.set_pos_perc(1000, 1)
        # AL_points = fetcher.get()
        # path = f'./active_learning/data/AL_n{n}_v1_trustScore_posOnly.csv'
        # fetcher.save(tr_df, AL_points, path)
        for k in [1000, 5000]:
            fetcher = MostCertain_SelectPoints(df, pos_perc=pos_perc)
            AL_points = fetcher.get()
            path = f'./active_learning/data/AL_n{n}_v1_topCertain_balanced_sample{k}.csv'
            fetcher.save(tr_df, AL_points, path)

            fetcher = MostCertain_SelectPoints(df, pos_perc=1)
            AL_points = fetcher.get()
            path = f'./active_learning/data/AL_n{n}_v1_topCertain_posOnly_sample{k}.csv'
            fetcher.save(tr_df, AL_points, path)

            fetcher = Random_SelectPoints(df)
            AL_points = fetcher.get()
            path = f'./active_learning/data/AL_n{n}_v1_random_balanced_sample{k}.csv'
            ipdb.set_trace() 
            fetcher.save(tr_df, AL_points, path)

            fetcher = RandomPosOnly_SelectPoints(df)
            AL_points = fetcher.get()
            path = f'./active_learning/data/AL_n{n}_v1_random_posOnly_sample{k}.csv'
            fetcher.save(tr_df, AL_points, path)


            fetcher = LeastCertain_SelectPoints(df, pos_perc=pos_perc)
            AL_points = fetcher.get()
            path = f'./active_learning/data/AL_n{n}_v1_leastCertain_balanced_sample{k}.csv'
            fetcher.save(tr_df, AL_points, path)

            fetcher = LeastCertain_SelectPoints(df, pos_perc=1)
            AL_points = fetcher.get()
            path = f'./active_learning/data/AL_n{n}_v1_leastCertain_posOnly_sample{k}.csv'
            fetcher.save(tr_df, AL_points, path)


