import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from statistics import median
import os
import riskslim


def LR_coef(data):
    #regression model
    lr_mod = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000,fit_intercept=False)
    lr_res = lr_mod.fit(data['X'], data['Y'].flatten())
    return(lr_res.coef_.flatten())

def round_coef(coef, alpha = 1.0):
    coef_new = coef.copy()/alpha
    coef_new = coef_new.round()
    coef_new[0] = coef[0]/alpha
    return(coef_new)

def get_metrics(data, coef, alpha = 1.0):

    # find v
    v = np.dot(alpha*data['X'], coef)
    v = v.astype(np.float128)
    v = np.clip(v, -709.78, 709.78)
    
    # get predicted probabilities
    prob_1 = 1.0 + np.exp(v)
    prob = np.exp(v)/(prob_1)
    pred_vals = np.zeros(shape=np.shape(prob))

    # get predicted values (0.5 cutoff)
    for i in range(len(prob)):
        if prob[i] >= 0.5:
            pred_vals[i] = 1
        else:
            pred_vals[i] = -1

    # AUC
    auc = roc_auc_score(data['Y'], prob)

    # Confusion matrix measures
    tn, fp, fn, tp = confusion_matrix(data['Y'], pred_vals, labels=[-1,1]).ravel()

    accuracy=(tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
            
    return auc, accuracy, sensitivity, specificity
    
def record_measures(f, n, p, method, measures, non_zer, res):
    data = [[f, n, p, method, measures[1], measures[2], measures[3], measures[0], non_zer]]
    new_row = pd.DataFrame(data=data, columns=column_names)
    return(pd.concat([res, new_row], ignore_index=True))

