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
column_names = ["data", "n", "p", "method", "acc", "sens", "spec", "auc", "lambda0", "non-zeros", "med_abs", "max_abs", "time"]



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
    
def record_measures(res, data, f, n, p, method, lambda0, coef, alpha, time):
    # Adds record
    measures = get_metrics(data, coef, alpha)

    data = [[f, n, p, method, measures[1], measures[2], measures[3], measures[0], lambda0, np.count_nonzero(coef[1:]), median(np.abs(coef[1:])), np.max(np.abs(coef[1:])), time]]
    new_row = pd.DataFrame(data=data, columns=column_names)
    return(pd.concat([res, new_row], ignore_index=True))


def run_experiments(my_path):
    # directory of files
    files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f))]

    res = pd.DataFrame(columns = column_names)
    #writer = pd.ExcelWriter(os.path.join(my_path,"res_coef_ncd.xlsx"), engine='openpyxl', mode='a',if_sheet_exists='replace')
    
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
        
        # ### NEW: Added train test split
        X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(data['X'], data['Y'], data["sample_weights"], test_size = 0.25, random_state = 42)
        
        #data_train = {'X': X_train, 'Y': y_train, 'sample_weights': weights_train}
        #data_test = {'X': X_test, 'Y': y_test, 'sample_weights': weights_test}
        data_train = {}
        data_test =  {}
        
        data_train["X"] = X_train
        data_train["Y"] = y_train
        data_train["sample_weights"] = weights_train
        data_test["X"] = X_test
        data_test["Y"] = y_test
        data_test["sample_weights"] = weights_test
        
        # X+y+weight (train) = data_train
        # same for test indicies
        
        # coefficient frame
        # coef_empty = np.zeros(shape=(p+1,3))
        # coef_df = pd.DataFrame(data=coef_empty, columns = ["Vars", "CPA","NCD"])
        # coef_df["Vars"] = data["variable_names"]
        data_train["variable_names"] = data["variable_names"]
        data_test["variable_names"] = data["variable_names"]
        


        lambda0 = np.logspace(-6, -1, 7)
        for i in range(len(lambda0)):
            
            # cpa
            s1 = time.time()
            coef_cpa = mtds.CPA_coef(data_train,lambda0[i])
            t1 = time.time()
            res = record_measures(res, data_test, f, n, p, "CPA", lambda0[i], coef_cpa, 1.0, t1-s1)
            #coef_df["CPA"] = coef_cpa
            
            # logistic regression
            s2 = time.time()
            coef_lr = mtds.LR_coef(data_train, weights_train.flatten())
            t2 = time.time()
            # res = record_measures(res, data, f, n, p, "LR", coef_lr, 1.0, t2-s2)
            #coef_df["LR"] = coef_lr

            # rounded logistic regression
            s3 = time.time()
            coef_rd = mtds.round_coef(coef_lr)
            t3 = time.time()
            # res = record_measures(res, data, f, n, p, "Round", coef_rd, 1.0, t2-s2+t3-s3)
            #coef_df["Round"] = coef_round
            
            # coordinate descent from LR
            coef_lr = mtds.LR_coef(data_train, weights_train.flatten())
            s5 = time.time()
            alpha = max(np.abs(coef_lr[1:]))/10.0 # bring within range
            coef_ncd = mtds.round_coef(coef_lr, alpha)
            ### NEW: changed cd to ncd.
            alpha_ncd, coef_ncd = ncd.coord_desc_nll(data_train, 1.0/alpha, coef_ncd, lambda0[i])
            t5 = time.time()
            res = record_measures(res, data_test, f, n, p, "NLLCD", lambda0[i], coef_ncd, alpha_ncd, t5-s5+t2-s2+t3-s3) # data_test
            #coef_df["NCD"] = coef_ncd
            
            # write coefficient info
            #coef_df.to_excel(writer, sheet_name=f, index=False)
            i += 1
        
        
    res.to_csv(os.path.join(my_path,"results_ncd.csv"), index=False)
    # writer.save()
    # writer.close()


if __name__ == "__main__":
    run_experiments(sys.argv[1])


# e-6 to e-=1