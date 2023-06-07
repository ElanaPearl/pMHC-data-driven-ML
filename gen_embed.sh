#python generate_embeddings.py -model_path './ckpt/AL_n10k_v1_random_posOnly_ckpt_e199.pth' -df_path './data/IEDB_regression_data.csv' -use_cuda 
#  .56, .28 - 10k
python generate_embeddings.py -model_path './ckpt/zesty-sea-186_ckpt_e139.pth' -df_path './data/IEDB_regression_data.csv' -use_cuda 
# 54, 27
python generate_embeddings.py -model_path './ckpt/AL_n10k_v1_topCertain_balanced_ckpt_e199.pth' -df_path './data/IEDB_regression_data.csv' -use_cuda 
