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


    data = riskslim.load_data_from_csv(dataset_csv_file = data_csv_file, sample_weights_csv_file = sample_weights_csv_file)
    
    return data['X'], data['Y'], data['variable_names']
# #--------------------- sample preprocess ---------------


X_breast, Y_breast, names_breast = LR_read_data('breastcancer')
X_spam, Y_spam, names_spam = LR_read_data('spam')
X_bank, Y_bank, names_bank = LR_read_data('bank')
X_tbrisk_cpa, Y_tbrisk_cpa, names_tbrisk_cpa = LR_read_data('tbrisk_cpa')
X_sim5_1, Y_sim5_1, names_sim5_1 = LR_read_data('simulate5_1_data')
X_sim5_2, Y_sim5_2, names_sim5_2 = LR_read_data('simulate5_2_data')
X_sim5_3, Y_sim5_3, names_sim5_3 = LR_read_data('simulate5_3_data')
X_sim5_4, Y_sim5_4, names_sim5_4 = LR_read_data('simulate5_4_data')
X_sim5_5, Y_sim5_5, names_sim5_5 = LR_read_data('simulate5_5_data')
X_sim5_6, Y_sim5_6, names_sim5_6 = LR_read_data('simulate5_6_data')
X_sim5_7, Y_sim5_7, names_sim5_7 = LR_read_data('simulate5_7_data')
X_sim5_8, Y_sim5_8, names_sim5_8 = LR_read_data('simulate5_8_data')
X_sim5_9, Y_sim5_9, names_sim5_9 = LR_read_data('simulate5_9_data')
X_sim5_10, Y_sim5_10, names_sim5_10 = LR_read_data('simulate5_10_data')
X_sim10_1, Y_sim10_1, names_sim10_1 = LR_read_data('simulate10_1_data')
X_sim10_2, Y_sim10_2, names_sim10_2 = LR_read_data('simulate10_2_data')
X_sim10_3, Y_sim10_3, names_sim10_3 = LR_read_data('simulate10_3_data')
X_sim10_4, Y_sim10_4, names_sim10_4 = LR_read_data('simulate10_4_data')
X_sim10_5, Y_sim10_5, names_sim10_5 = LR_read_data('simulate10_5_data')
X_sim10_6, Y_sim10_6, names_sim10_6 = LR_read_data('simulate10_6_data')
X_sim10_7, Y_sim10_7, names_sim10_7 = LR_read_data('simulate10_7_data')
X_sim10_8, Y_sim10_8, names_sim10_8 = LR_read_data('simulate10_8_data')
X_sim10_9, Y_sim10_9, names_sim10_9 = LR_read_data('simulate10_9_data')
X_sim10_10, Y_sim10_10, names_sim10_10 = LR_read_data('simulate10_10_data')
X_sim50_1, Y_sim50_1, names_sim50_1 = LR_read_data('simulate50_1_data')
X_sim50_2, Y_sim50_2, names_sim50_2 = LR_read_data('simulate50_2_data')
X_sim50_3, Y_sim50_3, names_sim50_3 = LR_read_data('simulate50_3_data')
X_sim50_4, Y_sim50_4, names_sim50_4 = LR_read_data('simulate50_4_data')
X_sim50_5, Y_sim50_5, names_sim50_5 = LR_read_data('simulate50_5_data')
X_sim50_6, Y_sim50_6, names_sim50_6 = LR_read_data('simulate50_6_data')
X_sim50_7, Y_sim50_7, names_sim50_7 = LR_read_data('simulate50_7_data')
X_sim50_8, Y_sim50_8, names_sim50_8 = LR_read_data('simulate50_8_data')
X_sim50_9, Y_sim50_9, names_sim50_9 = LR_read_data('simulate50_9_data')
X_sim50_10, Y_sim50_10, names_sim50_10 = LR_read_data('simulate50_10_data')


# #------------ LR function returns 2 coefs vectors --------------

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
    cm = confusion_matrix(y_test, y_pred)
    # print('Confusion Matrix : \n', cm)

    total=sum(sum(cm))
    #####from confusion matrix calculate accuracy
    accuracy=(cm[0,0]+cm[1,1])/total
    # print ('Accuracy : ', accuracy)

    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    # print('Sensitivity : ', sensitivity)

    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    # print('Specificity : ', specificity)

    return lr_coef, coef_round, lr_int, y_pred_prob, y_pred_prob, auc, accuracy, sensitivity, specificity


#---------sample LR ---------------
lr_coef_, coef_round, lr_int, y_pred_prob, y_pred_prob, _, _, _, _= LR_func(X_breast, Y_breast)

for i in len(list_X):


#----------- run LR_func and get results from it for 34 datasets

file_name = ['breastcancer', 'spam', 'bank', 'tbrisk_cpa','simulate5_1_data', 'simulate5_2_data', 'simulate5_3_data', 
             'simulate5_4_data', 'simulate5_5_data', 'simulate5_6_data', 'simulate5_7_data', 'simulate5_8_data', 
             'simulate5_9_data', 'simulate5_10_data', 'simulate10_1_data', 'simulate10_2_data','simulate10_3_data', 
             'simulate10_4_data', 'simulate10_5_data', 'simulate10_6_data', 'simulate10_7_data', 'simulate10_8_data', 
             'simulate10_9_data', 'simulate10_10_data', 'simulate50_1_data', 'simulate50_2_data', 'simulate50_3_data', 'simulate50_4_data', 'simulate50_5_data', 'simulate50_6_data', 'simulate50_7_data', 'simulate50_8_data', 'simulate50_9_data', 'simulate50_10_data']

def LR_process()