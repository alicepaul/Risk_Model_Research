import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import os
import pysal
import spreg
from scipy.stats import chi2, norm


#data
data_name = "breastcancer"                                  # name of the data
data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
data_csv_file = data_dir + data_name + '_data.csv'
feature_names = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']
column_names = ['Benign'] + feature_names
data = pd.read_csv(data_csv_file,names=column_names)

#data cleaning
data = data.replace(to_replace='?',value=np.nan)    
data = data.dropna(how='any')      
data = data.iloc[1: , :]

print(data.shape)

X = data.iloc[:,1:10]
y = data.iloc[:,[0]]

print(X, y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#regression model
lr = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000).fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)[:,1]
y_pred = lr.predict(X_test)



# accuracy, f1
report = classification_report(y_test, y_pred)
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_pred_prob)
print('auc', auc)

#specificity, sensitivity
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix : \n', cm)

total=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy=(cm[0,0]+cm[1,1])/total
print ('Accuracy : ', accuracy)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity)

specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity)




# save coef
coef_lr_df = pd.DataFrame(list(X.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr.coef_.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)
coef_lr_df.loc[-1] = ('Intercept', lr.intercept_.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# coef_lr_df.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/lr_breastcancer_unrounded.csv')
print('----------------- Coef unrounded -------------')
print(coef_lr_df)

coef_lr_df['Coefs'] = coef_lr_df['Coefs'].astype(float)
coef_lr_df['Coefs'].loc[1:9] = (coef_lr_df['Coefs'].loc[1:9] * 2).round(0)
print('----------------- Coef * 2 rounded -------------')
print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]
# filter_coef_lr.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_lr_breastcancer.csv')
with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare.xlsx',
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='filter_lr_breastcancer')  