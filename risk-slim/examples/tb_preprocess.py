import pandas as pd
import numpy as np
import os

#data
data_name = "tbrisk"                                  # name of the data
data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
data_csv_file = data_dir + data_name + '_data.csv'
df = pd.read_csv(data_csv_file)

df = df.replace(to_replace='NA',value=np.nan)    
df = df.dropna(how='any')
# df['tb'] = np.where(df['tb']==2, 0, df['tb'])

# df.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/data/tbrisk_cpa_data.csv',index=False)



conditions = [
    (df['age_group'] == '[15,25)'),
    (df['age_group'] == '[25,35)'),
    (df['age_group'] == '[35,45)'),
    (df['age_group'] == '[45,55)'),
    (df['age_group'] == '[55,99)')
    ]

# create a list of the values we want to assign for each condition
values_1 = ['1', '0', '0', '0', '0']
values_2 = ['0', '1', '0', '0', '0']
values_3= ['0', '0', '1', '0', '0']
values_4= ['0', '0', '0', '1', '0']
values_5= ['0', '0', '0', '0', '1']

# create a new column and use np.select to assign values to it using our lists as arguments
df['age_1525'] = np.select(conditions, values_1)
df['age_2535'] = np.select(conditions, values_2)
df['age_3545'] = np.select(conditions, values_3)
df['age_4555'] = np.select(conditions, values_4)
df['age_5599'] = np.select(conditions, values_5)

df = df.drop(['age_group'], axis=1)

df.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/data/tbrisk_cpa_data.csv',index=False)
