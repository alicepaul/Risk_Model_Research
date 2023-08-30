import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from statistics import median
import methods as mtds
import NLL_coordinate_descent as ncd
import os
import riskslim
import time
import sys


# results dataframe and excel sheets
column_names = "data, n, p, method, acc, sens, spec, auc, lambda0, non-zeros, med_abs, max_abs, time \n"

def get_metrics(data, coef, alpha = 1.0):
    
    # replace -inf in coef to approximate real number 
    coef = np.nan_to_num(coef)
    
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
    
def record_measures(data, f, n, p, method, lambda0, coef, alpha, time):
    # Adds record
    measures = get_metrics(data, coef, alpha)

    res_str = f + "," + str(n) + "," + str(p) + "," + method + "," + str(measures[1]) + "," + str(measures[2]) + "," \
    + str( measures[3]) + "," + str(measures[0])+"," + str(lambda0) + "," + str(np.count_nonzero(coef[1:]))+ "," \
    + str(median(np.abs(coef[1:])))+","+str(np.max(np.abs(coef[1:]))) + ","+str( time) +"\n"
    return(res_str)


def run_experiments(my_path):
    # directory of files
    files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f))]

    #res = pd.DataFrame(columns = column_names)
    res_file = os.path.join(my_path,"results_ncd_fr.csv")
    res_f = open(res_file, "w")
    res_f.write(column_names)
    res_f.close()
    
    # iterate through files
    i = 1
    for f in files:

        # only do data files
        if "_data.csv" not in f:
            continue
        
        
        # read data and add weight file if applicable
        print(f)
        print("Iteration: ", i, " of ", len(files))
        sample_f =  os.path.join(my_path, f[0:(len(f)-8)]+"weights.csv")
        if (os.path.isfile(sample_f) == False):
            sample_f = None
        data = riskslim.load_data_from_csv(dataset_csv_file = os.path.join(my_path,f),
                                       sample_weights_csv_file = sample_f)
        n = data["X"].shape[0]  #rows
        p = data["X"].shape[1]-1
        weights = data["sample_weights"]
        print(n,p)
        
        # test train split
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(data['X'], data['Y'], data["sample_weights"], test_size = 0.25, random_state = 42)
        data_train = {}
        data_test =  {}
        data_train["X"] = X_train
        data_train["Y"] = y_train
        data_train["sample_weights"] = weights_train
        data_test["X"] = X_test
        data_test["Y"] = y_test
        data_test["sample_weights"] = weights_test
        data_train["variable_names"] = data["variable_names"]
        data_test["variable_names"] = data["variable_names"]
        


        #lambda0 = np.logspace(-6, -1, 7)
        #for i in range(len(lambda0)):
        res = ""
            
        # cpa
        #s1 = time.time()
        #coef_cpa = mtds.CPA_coef(data_train)
        #t1 = time.time()
        #res += record_measures(data_test, f, n, p, "CPA", 0, coef_cpa, 1.0, t1-s1)

        # fr
        s1 = time.time()
        coef_cpa = mtds.FR_coef(data_train)
        t1 = time.time()
        res += record_measures(data_test, f, n, p, "FR", 0, coef_cpa, 1.0, t1-s1)
            
        # logistic regression
        #s2 = time.time()
        #coef_lr = mtds.LR_coef(data_train, weights_train.flatten())
        #t2 = time.time()
        #res += record_measures(data_test, f, n, p, "LR", 0, coef_lr, 1.0, t2-s2)

        # rounded logistic regression
        #s3 = time.time()
        #coef_rd = mtds.round_coef(coef_lr)
        #t3 = time.time()
        #res += record_measures(data_test, f, n, p, "Round", 0, coef_rd, 1.0, t2-s2+t3-s3)
            
        # coordinate descent from LR
        #coef_lr = mtds.LR_coef(data_train, weights_train.flatten())
        #s5 = time.time()
        #alpha = max(np.abs(coef_lr[1:]))/10.0 # bring within range
        #coef_ncd = mtds.round_coef(coef_lr, alpha)
        #alpha_ncd, coef_ncd = ncd.coord_desc_nll(data_train, 1.0/alpha, coef_ncd)
        #t5 = time.time()
        #res += record_measures(data_test, f, n, p, "NLLCD", 0, coef_ncd, alpha_ncd, t5-s5+t2-s2) # data_test

        res_f = open(res_file, "a")
        res_f.write(res)
        res_f.close()
        



if __name__ == "__main__":
    run_experiments(sys.argv[1])


# e-6 to e-=1
