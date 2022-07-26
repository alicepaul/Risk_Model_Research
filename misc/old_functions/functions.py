import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import os
import pprint
import riskslim
import pickle

# -------------- Preprocess tb risk data -------------------
# #data
# data_name = "tbrisk"                                  # name of the data
# data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
# data_csv_file = data_dir + data_name + '_data.csv'
# data = pd.read_csv(data_csv_file)


# data_np = data.to_numpy()

# #extract vector Y of tb
# Y_tb = data_np[:,[0]]

# dummy = np.zeros(shape = (np.shape(data_np)[0],5))


# for i in range(np.shape(data_np)[0]):
#     # level A
#     if data_np[i,1] =='[15,25)':
#         dummy[i] = np.array([1,0,0,0,0])
#     # level B
#     elif data_np[i,1] =='[25,35)':
#         dummy[i] = np.array([0,1,0,0,0])
#     # level C
#     elif data_np[i,1] =='[35,45)':
#         dummy[i] = np.array([0,0,1,0,0])
#     # level D
#     elif data_np[i,1] =='[45,55)':
#         dummy[i] = np.array([0,0,0,1,0])
#     # level E
#     else:
#         dummy[i] = np.array([0,0,0,0,1])
        
# data_np = np.concatenate((data_np,dummy),axis=1)
    
# M_tb = data_np[:,1:]

# M_tb = M_tb.astype(np.float32)

# Y_tb = Y_tb.flatten()
# Y_tb = Y_tb.astype(int)

#------------- Prepocess other dataset using numpy, need to add variables names at end -----------------

# def data_preprocess(data_name):
#     #data                                 
#     # data_name = 'tbrisk_cpa'                                  # name of the data
#     data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
#     data_csv_file = data_dir + data_name + '_data.csv'
#     data = pd.read_csv(data_csv_file)

#     # data cleaning
#     data = data.replace(to_replace='NA',value=np.nan)    
#     data = data.dropna(how='any')

#     data_np = data.to_numpy()

#     #extract vector Y of tb
#     Y = data_np[:,[0]]
#     Y = Y.flatten()
#     Y = Y.astype(int)
    
#     M = data_np[:,1:]
#     M = M.astype(np.float32)

#     return Y, M


#----------------------- try CPA data reading --------------
def LR_read_data(data_name):
    # data_name = "breastcancer"                                  # name of the data
    data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
    data_csv_file = data_dir + data_name + '_data.csv'          # csv file for the dataset
    sample_weights_csv_file = None                              # csv file of sample weights for the dataset (optional)
    df = pd.read_csv(data_csv_file)

    #define target variable
    X = df.iloc[:,1:]
    y = df.iloc[:,[0]]

    data = riskslim.load_data_from_csv(dataset_csv_file = data_csv_file, sample_weights_csv_file = sample_weights_csv_file)
    return data['X'], data['Y'], data['variable_names'], df

#------------ LR function returns 2 coefs vectors --------------

def LR_func(X,Y):
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    #regression model
    lr = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000).fit(X_train, y_train)
    y_pred_prob = lr.predict_proba(X_test)[::,1]
    y_pred = lr.predict(X_test)


    #coef with round to nearest integer, alpha=1
    alpha = 1
    lr_coef = lr.coef_ #without rounding
    coef_round = lr_coef.round(0)
    lr_int = lr.intercept_
    
    # model eval
    report = classification_report(y_test, y_pred)
    # print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_pred_prob)
    # print('auc', auc)

    #specificity, sensitivity
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    # print('Confusion Matrix : \n', cm)

    total=sum(sum(cm))
    #####from confusion matrix calculate accuracy
    accuracy=(cm[0,0]+cm[1,1])/total
    # print ('Accuracy : ', accuracy)

    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    # print('Sensitivity : ', sensitivity)

    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    # print('Specificity : ', specificity)

    return lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity


#------------------ 34 datasets result computation ----------------------
# #--------------------- sample preprocess ---------------

# X_breast, Y_breast, names_breast, df_breast = LR_read_data('breastcancer')


#------------------ change last line to create or append to xlsx ----------------------------------------

# lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_breast, Y_breast)

# coef_lr_df = pd.DataFrame(list(df_breast.columns)).copy()
# coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
# coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# # print('----------------- Coef unrounded -------------')
# # print(coef_lr_df)

# coef_lr_df['Coefs'] = coef_round.transpose()
# coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
# coef_lr_df.index = coef_lr_df.index + 1  # shifting index
# coef_lr_df.sort_index(inplace=True)
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
#                 pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
#                 pd.Series(['AUC', auc], index=coef_lr_df.columns )]
# coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# # print('----------------- Coef rounded -------------')
# # print(coef_lr_df)
# filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

# print(accuracy)



# # #--------------------------------
# lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_tbrisk_cpa, Y_tbrisk_cpa)

# coef_lr_df = pd.DataFrame(list(df_tb.columns)).copy()
# coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
# coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# # print('----------------- Coef unrounded -------------')
# # print(coef_lr_df)

# coef_lr_df['Coefs'] = coef_round.transpose()
# coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
# coef_lr_df.index = coef_lr_df.index + 1  # shifting index
# coef_lr_df.sort_index(inplace=True)
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
#                 pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
#                 pd.Series(['AUC', auc], index=coef_lr_df.columns )]
# coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# # print('----------------- Coef rounded -------------')
# # print(coef_lr_df)
# filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# # filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

# with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
#                     mode='a') as writer:  
#     filter_coef_lr.to_excel(writer, sheet_name='tb.dat.LR_')  

# print(accuracy)

# # #-------------------


# lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_spam, Y_spam)

# coef_lr_df = pd.DataFrame(list(df_spam.columns)).copy()
# coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
# coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# # print('----------------- Coef unrounded -------------')
# # print(coef_lr_df)

# coef_lr_df['Coefs'] = coef_round.transpose()
# coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
# coef_lr_df.index = coef_lr_df.index + 1  # shifting index
# coef_lr_df.sort_index(inplace=True)
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
#                 pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
#                 pd.Series(['AUC', auc], index=coef_lr_df.columns )]
# coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# # print('----------------- Coef rounded -------------')
# # print(coef_lr_df)
# filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# # filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

# with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
#                     mode='a') as writer:  
#     filter_coef_lr.to_excel(writer, sheet_name='spam.dat.LR_')  
# print(accuracy)


# # -------------------------





# lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_bank, Y_bank)

# coef_lr_df = pd.DataFrame(list(df_bank.columns)).copy()
# coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
# coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# # print('----------------- Coef unrounded -------------')
# # print(coef_lr_df)

# coef_lr_df['Coefs'] = coef_round.transpose()
# coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
# coef_lr_df.index = coef_lr_df.index + 1  # shifting index
# coef_lr_df.sort_index(inplace=True)
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
#                 pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
#                 pd.Series(['AUC', auc], index=coef_lr_df.columns )]
# coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# # print('----------------- Coef rounded -------------')
# # print(coef_lr_df)
# filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# # filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

# with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
#                     mode='a') as writer:  
#     filter_coef_lr.to_excel(writer, sheet_name='bank.dat.LR_')  

# print(accuracy)





#--------------- simulate data -----------
#-----------------LR---------------------

#---------------------------------------------

lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_1, Y_sim50_1)


coef_lr_df = pd.DataFrame(list(df_50_1.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.9106')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_1.dat.LR_')  

print(accuracy)


#---------------------------------------------

lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_2, Y_sim50_2)


coef_lr_df = pd.DataFrame(list(df_50_2.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_2.dat.LR_')  

print(accuracy)



#---------------------------------------------

lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_3, Y_sim50_3)


coef_lr_df = pd.DataFrame(list(df_50_3.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_3.dat.LR_')  

print(accuracy)




#---------------------------------------------

lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_4, Y_sim50_4)


coef_lr_df = pd.DataFrame(list(df_50_4.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_4.dat.LR_')  

print(accuracy)



#---------------------------------------------

lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_5, Y_sim50_5)


coef_lr_df = pd.DataFrame(list(df_50_5.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_5.dat.LR_')  

print(accuracy)


#-------------------------------------

lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_6, Y_sim50_6)


coef_lr_df = pd.DataFrame(list(df_50_6.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_6.dat.LR_')  

print(accuracy)

#---------------------------------------



lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_7, Y_sim50_7)


coef_lr_df = pd.DataFrame(list(df_50_7.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_7.dat.LR_')  

print(accuracy)

#---------------------------------------


lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_8, Y_sim50_8)


coef_lr_df = pd.DataFrame(list(df_50_8.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_8.dat.LR_')  

print(accuracy)

#---------------------------------------



lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_9, Y_sim50_9)


coef_lr_df = pd.DataFrame(list(df_50_9.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_9.dat.LR_')  

print(accuracy)

#---------------------------------------



lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X_sim50_10, Y_sim50_10)


coef_lr_df = pd.DataFrame(list(df_50_10.columns)).copy()
coef_lr_df.insert(len(coef_lr_df.columns),"Coefs",lr_coef.transpose())
coef_lr_df.rename(columns = {0:'Features'}, inplace = True)

# print('----------------- Coef unrounded -------------')
# print(coef_lr_df)

coef_lr_df['Coefs'] = coef_round.transpose()
coef_lr_df.loc[-1] = ('Intercept', lr_int.transpose())
coef_lr_df.index = coef_lr_df.index + 1  # shifting index
coef_lr_df.sort_index(inplace=True)
listOfSeries = [pd.Series(['Accuracy', accuracy], index=coef_lr_df.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=coef_lr_df.columns ) ,
                pd.Series(['Specificity', specificity], index=coef_lr_df.columns ),
                pd.Series(['AUC', auc], index=coef_lr_df.columns )]
coef_lr_df = coef_lr_df.append(listOfSeries,ignore_index=True)
# print('----------------- Coef rounded -------------')
# print(coef_lr_df)
filter_coef_lr = coef_lr_df[(coef_lr_df['Coefs'] != 0.0) & (coef_lr_df['Coefs'] !=-0.0) ]

# filter_coef_lr.to_excel("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx", sheet_name='breast.dat.LR_0.956')

with pd.ExcelWriter("/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx",
                    mode='a') as writer:  
    filter_coef_lr.to_excel(writer, sheet_name='sim50_10.dat.LR_')  

print(accuracy)

#---------------------------------------


