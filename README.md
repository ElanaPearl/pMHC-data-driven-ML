# pMHC-data-driven-ML

### Getting Started
* `conda env create  -f environment.yml; conda activate cs273` to create and activate environment
* `python scripts/download_data.py` to download the training dataset


## Training the model 
The model can be trained by running `python main.py {args}`. The recommended run is `python main.py -device 0` where the recommended hyperparameters are by default. 

### Data
By default, the model is trained on all 5 splits of the classification single allele data. This is ~4ish million data points with (noisy) binary labels. Only 5% positives. 

The validation data during training is split = 4 from the high quality regression data (so <50k points). Splits 0,1,2,3 from regression are the test data. 
### Hyperparameter notes and training recommendations
* `-device` sepecify cuda device, default: 0.
 <!-- necessecity / strongly recommended.  -->
* Model hyperparameters are already tuned (ie. learning rate, model params, etc.) and are recommended to leave as they are. Set as default. 
* The only version of the model working is the simple LSTM version, taken from the MHCAttNet paper and codebase (see [here](https://github.com/gopuvenkat/MHCAttnNet/tree/master)). 
    * This model takes as input the index representation of the AA strings (ie. so 'AAC' -> '1,1,2'). 
    * 1-indexed, since '0' is the padding index. 
* Model states are saved under `-save_path` which is by default the `ckpt/` folder. 
* Metrics are logged in wandb. 

### Generating embeddings 
Since the model runs for an indeterminate set of time, and generating all classifciation embeddings takes >20 min, I decided to have the embedding generation process run in a separate file. Run `python scripts/generate_embeddings.py -device 0 -model_path {...} -df_path {...}`.
* `-model_path` - where the model weights are stowed. For example, './ckpt/sage-haze-125_ckpt_e37.pth'.
* `-df_path` - what dataset to generate embeddings for. For example, for classification, './data/IEDB_classification_data_SA.csv'. 
* `-save_path` (optional) - by default, will save the embeddings under './embeds/{wandb run name}_{clssfctn or rgrssn}.csv'.

The output of running this script the same dataframe as in `df_path` but with 3 new column information:
* pred_affinity - predicted normalized affinity score from the model
* pred_affinity_logits - predicted logit affinity score from the model
* latents_{i} - takes the penultimate layer of the model (typically dim 64) and saves all the latents


## Retrain model using training instances w/ high confidence

E.g., taking the top 90% instances (per class) with the highest confidence:
```
python main.py -device 0 -model_path /dfs/user/shirwu/course/cs273b/pMHC-data-driven-ML/ckpt/floral-snowflake-41_ckpt_e2.pth -threshold 0.9
```