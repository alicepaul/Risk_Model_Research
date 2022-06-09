import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os



#data
data_name = "spambase"                                  # name of the data
data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
data_csv_file = data_dir + data_name + '_data.csv'
data = pd.read_csv(data_csv_file)

#data cleaning
data = data.replace(to_replace='?',value=np.nan)    
data = data.dropna(how='any')      

print(data.shape)

#define target variable
X = data.iloc[:,1:58]
y = data.iloc[:,[0]]


#regression model
lr1 = LogisticRegression(penalty="l1",solver="liblinear",C=0.8,max_iter=1000)
lr1.fit(X,y)

lr2 = LogisticRegression(penalty="l2",solver="lbfgs",C=0.8,max_iter=1000)
lr2.fit(X,y)


# save coef
coef_l1_df = pd.DataFrame(list(X.columns)).copy()
coef_l1_df.insert(len(coef_l1_df.columns),"Coefs",lr1.coef_.transpose())
coef_l1_df.rename(columns = {0:'Features'}, inplace = True)
coef_l1_df.loc[-1] = ('Intercept', lr1.intercept_.transpose())
coef_l1_df.index = coef_l1_df.index + 1  # shifting index
coef_l1_df.sort_index(inplace=True) 

coef_l1_df['Coefs'] = coef_l1_df['Coefs'].astype(float).round(0)
print(coef_l1_df)
coef_l1_df.to_csv('/Users/zhaotongtong/Desktop/Summer_Research_2022/risk-slim/examples/results/lr1_spam.csv')
filter_coef_l1 = coef_l1_df[(coef_l1_df['Coefs'] != 0.0) & (coef_l1_df['Coefs'] !=-0.0) ]
filter_coef_l1.to_csv('/Users/zhaotongtong/Desktop/Summer_Research_2022/risk-slim/examples/results/filter_lr1_spam.csv')
# print((lrl1.coef_ != 0).sum(axis=1))

print('----------------------l2 coef below-----------------------------------------')
coef_l2_df = pd.DataFrame(list(X.columns)).copy()
coef_l2_df.insert(len(coef_l2_df.columns),"Coefs",lr2.coef_.transpose())
coef_l2_df.rename(columns = {0:'Features'}, inplace = True)
coef_l2_df.loc[-1] = ['Intercept', lr2.intercept_.transpose()]
coef_l2_df.index = coef_l2_df.index + 1  # shifting index
coef_l2_df.sort_index(inplace=True) 

coef_l2_df['Coefs'] = coef_l2_df['Coefs'].astype(float).round(0)
print(coef_l2_df)
coef_l2_df.to_csv('/Users/zhaotongtong/Desktop/Summer_Research_2022/risk-slim/examples/results/lr2_spam.csv')
filter_coef_l2 = coef_l1_df[(coef_l2_df['Coefs'] != 0.0) & (coef_l2_df['Coefs'] !=-0.0) ]
filter_coef_l2.to_csv('/Users/zhaotongtong/Desktop/Summer_Research_2022/risk-slim/examples/results/filter_lr2_spam.csv')