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



#------------- Prepocess other dataset -----------------

def data_preprocess(data_name):
    #data                                 # name of the data
    data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
    data_csv_file = data_dir + data_name + '_data.csv'
    data = pd.read_csv(data_csv_file)

    # data cleaning
    data = data.replace(to_replace='NA',value=np.nan)    
    data = data.dropna(how='any')

    data_np = data.to_numpy()

    #extract vector Y of tb
    Y = data_np[:,[0]]
    Y = Y.flatten()
    Y = Y.astype(int)
    
    M = data_np[:,1:]
    M = M.astype(np.float32)

    return Y, M


# -------------- Preprocess tb risk data -------------------
#data
data_name = "tbrisk"                                  # name of the data
data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
data_csv_file = data_dir + data_name + '_data.csv'
data = pd.read_csv(data_csv_file)

#data cleaning
# data = data.replace(to_replace='NA',value=np.nan)    
# data = data.dropna(how='any')
# data['tb'] = np.where(data['tb']==2, 0, data['tb'])


data_np = data.to_numpy()

#extract vector Y of tb
Y_tb = data_np[:,[0]]

dummy = np.zeros(shape = (np.shape(data_np)[0],5))


for i in range(np.shape(data_np)[0]):
    # level A
    if data_np[i,1] =='[15,25)':
        dummy[i] = np.array([1,0,0,0,0])
    # level B
    elif data_np[i,1] =='[25,35)':
        dummy[i] = np.array([0,1,0,0,0])
    # level C
    elif data_np[i,1] =='[35,45)':
        dummy[i] = np.array([0,0,1,0,0])
    # level D
    elif data_np[i,1] =='[45,55)':
        dummy[i] = np.array([0,0,0,1,0])
    # level E
    else:
        dummy[i] = np.array([0,0,0,0,1])
        
data_np = np.concatenate((data_np,dummy),axis=1)
    
M_tb = data_np[:,1:]

M_tb = M_tb.astype(np.float32)

Y_tb = Y_tb.flatten()
Y_tb = Y_tb.astype(int)


# #--------------------- sample preprocess ---------------

Y_breast, M_breast = data_preprocess('breastcancer')

# !!!! need to finish with other datasets 30+2


# #------------ LR function returns 2 coefs vectors --------------

def LR_func(M,Y):
    X_train,X_test,y_train,y_test = train_test_split(M_breast,Y_breast,test_size=0.2,random_state=42)

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

    return lr_coef, coef_round, lr_int, y_pred_prob, y_pred, auc, accuracy, sensitivity, specificity


#---------sample LR ---------------
lr_coef, coef_round, lr_int, _, _, _, _, _, _= LR_func(M_breast, Y_breast)

print("no rounding",lr_coef)
print("rounding", coef_round)
print("int", lr_int)


