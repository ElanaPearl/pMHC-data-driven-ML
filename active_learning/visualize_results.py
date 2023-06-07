import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
sns.set_theme()
# ipdb> data = pd.concat([data, pd.DataFrame({'model': ['10k_baseline'], 'aucroc': [.54], 'aucprc':[.27], 'base_n':['10k'], 'n_added': [0], 'how_added': ['NA']})], ignore_index=True)
metric_dict = {'trustScore_leastcertain_balanced':'Lowest Trust Score (B)',
'trustScore_mostcertain_balanced':'Highest Trust Score (B)',
'trustScore_mostcertain_posOnly':'Highest Trust Score (+)',
'random_balanced': 'Random (B)',
'random_posOnly': 'Random (+)',
'leastCertain_balanced':'Highest Entropy (B)',
'topCertain_balanced':'Lowest Entropy (B)',
'topCertain_posOnly':'Lowest Entropy (+)'}
if __name__ == '__main__': 
    base = '5k'
    
    if base == '10k':
        data =pd.read_csv('results_df.csv')
        colors = ['#001219', '#005F73', '#0A9396', '#94D2BD', '#E9D8A6', '#EE9B00', '#CA6702', '#BB3E03', '#AE2012', '#9B2226']
        data = data[data.n_added != 0]
        data = data[data.n_added != '1000.csv']
        data = data.sort_values(by='aucroc', ascending=False)
        data = data.drop([25, 23])
        data['how_added'] = data.apply(lambda x: metric_dict[x.how_added[1:-1]], axis=1)
        for auc, baseline in zip(['aucroc', 'aucprc'], [.54, .27]):
            import matplotlib.pyplot as plt 
            plt.subplots(figsize=(12,5), dpi=100)
            sns.barplot(x='n_added', y=auc, hue='how_added', data=data, palette=colors)
            plt.axhline(y=baseline, color='black', linestyle='--', label='Training Baseline (on 10k points)')
            plt.title(f'Active Learning: Effect on {auc.upper()} Performance\nfrom Adding Points To 10k Dataset',fontsize=12)
            plt.xlabel('Number of Points Added')
            plt.ylabel(f'{auc.upper()} on Heldout Regression Data')
            plt.legend(title="Uncertainty Metric", loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()


            plt.savefig(f'{auc}.png')
            plt.clf()
    else:
        data =pd.read_csv('results_df_5k.csv') 
        colors = ['#001219', '#005F73', '#0A9396', '#94D2BD', '#E9D8A6', '#EE9B00', '#CA6702', '#BB3E03', '#AE2012', '#9B2226']
        data = data.sort_values(by='aucroc', ascending=False)
        import ipdb; ipdb.set_trace()
        drop_models = ['./ckpt/AL_n5k_v1_trustScore_mostcertain_balanced_sample10000_ckpt_e199.pth', 
                        './ckpt/AL_n5k_v1_trustScore_leastcertain_balanced_sample10000_ckpt_e199.pth',
                        './ckpt/AL_n5k_v1_topCertain_posOnly_sample10000_ckpt_e199.pth',
                        './ckpt/AL_n5k_v1_topCertain_balanced_sample10000_ckpt_e199.pth',
                        './ckpt/AL_n5k_v1_topCertain_balanced_sample5000_ckpt_e199.pth',
                        './ckpt/AL_n5k_v1_leastCertain_balanced_sample5000_ckpt_e199.pth',
                        './ckpt/AL_seed41_n5k_v1_topCertain_posOnly_sample10000_ckpt_e199.pth',
                        './ckpt/AL_n5k_v1_trustScore_leastcertain_balanced_sample5000_ckpt_e199.pth',
                        './ckpt/AL_seed41_n5k_v1_trustScore_leastcertain_balanced_sample1000_ckpt_e199.pth',
                        './ckpt/AL_seed41_n5k_v1_trustScore_leastcertain_balanced_sample10000_ckpt_e199.pth',
                        './ckpt/AL_seed43_pw_n5k_v1_trustScore_leastcertain_balanced_sample10000_ckpt_e199.pth',
                        './ckpt/AL_pw1_n5k_v1_trustScore_leastcertain_balanced_sample1000_ckpt_e199.pth',
                        './ckpt/AL_seed43_pw20_n5k_v1_trustScore_mostcertain_balanced_sample10000_ckpt_e199.pth',
                        './ckpt/AL_seed41_n5k_v1_trustScore_mostcertain_balanced_sample10000_ckpt_e199.pth',
                        ]
        data = data[~data.model.isin(drop_models)]
        data['how_added'] = data.apply(lambda x: metric_dict[x.how_added[1:-1]], axis=1)
        for auc, baseline in zip(['aucroc', 'aucprc'], [.52, .26]):
            import matplotlib.pyplot as plt 
            plt.subplots(figsize=(12,5), dpi=100)
            sns.barplot(x='n_added', y=auc, hue='how_added', data=data, palette=colors)
            plt.axhline(y=baseline, color='black', linestyle='--', label='Training Baseline (on 3k points)')
            plt.title(f'Active Learning: Effect on {auc.upper()} Performance\nfrom Adding Points To 3k Dataset',fontsize=12)
            plt.xlabel('Number of Points Added')
            plt.ylabel(f'{auc.upper()} on Heldout Regression Data')
            plt.legend(title="Uncertainty Metric", loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()


            plt.savefig(f'{auc}_5k.png')
            plt.clf()