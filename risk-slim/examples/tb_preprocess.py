import pandas as pd
import numpy as np
import os

#data
data_name = "tbrisk"                                  # name of the data
data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
data_csv_file = data_dir + data_name + '_data.csv'
data = pd.read_csv(data_csv_file)

data = data.replace(to_replace='NA',value=np.nan)    
data = data.dropna(how='any')
data['tb'] = np.where(data['tb']==2, 0, data['tb'])
data = data.drop(['age_group'], axis=1)
data.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/data/tbrisk_data.csv',index=False)