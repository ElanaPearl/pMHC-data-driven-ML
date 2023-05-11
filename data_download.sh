# Script that downloads and reformats the NetMHCpan training data from the IEDB
echo "Downloading data..."
wget https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz -O NetMHCpan_train_download.tar.gz
echo "Unzipping..."
tar  -xvzf NetMHCpan_train_download.tar.gz
echo "Reformatting..."
python reformat_data_download.py
rm NetMHCpan_train_download.tar.gz
rm -r NetMHCpan_train/
echo "Done! Data saved at data/IEDB_regression_data.csv"