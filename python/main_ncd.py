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
#import riskslim
import time 
import sys
from pathlib import Path
import warnings


# results dataframe and excel sheets
column_names = "data, n, p, method, acc, sens, spec, auc, lambda0, non-zeros, med_abs, max_abs, time \n"

def get_metrics(data, coef, alpha = 1.0):
    
    # replace -inf in coef to approximate real number 
    coef = np.nan_to_num(coef)
    v = np.dot(alpha*data['X'], coef)
    v = v.astype(np.float64)
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

def check_data(data):

    # type checks
    assert type(data) is dict, "data should be a dict"

    assert 'X' in data, "data should contain X matrix"
    assert type(data['X']) is np.ndarray, "type(X) should be numpy.ndarray"

    assert 'Y' in data, "data should contain Y matrix"
    assert type(data['Y']) is np.ndarray, "type(Y) should be numpy.ndarray"

    assert 'variable_names' in data, "data should contain variable_names"
    assert type(data['variable_names']) is list, "variable_names should be a list"

    X = data['X']
    Y = data['Y']
    variable_names = data['variable_names']

    if 'outcome_name' in data:
        assert type(data['outcome_name']) is str, "outcome_name should be a str"

    # sizes and uniqueness
    N, P = X.shape
    assert N > 0, 'X matrix must have at least 1 row'
    assert P > 0, 'X matrix must have at least 1 column'
    assert len(Y) == N, 'dimension mismatch. Y must contain as many entries as X. Need len(Y) = N.'
    assert len(list(set(data['variable_names']))) == len(data['variable_names']), 'variable_names is not unique'
    assert len(data['variable_names']) == P, 'len(variable_names) should be same as # of cols in X'

    # feature matrix
    assert np.all(~np.isnan(X)), 'X has nan entries'
    assert np.all(~np.isinf(X)), 'X has inf entries'

    # offset in feature matrix
    if '(Intercept)' in variable_names:
        assert all(X[:, variable_names.index('(Intercept)')] == 1.0), "(Intercept)' column should only be composed of 1s"
    else:
        warnings.warn("there is no column named INTERCEPT_NAME in variable_names")

    # labels values
    assert all((Y == 1) | (Y == -1)), 'Need Y[i] = [-1,1] for all i.'
    if all(Y == 1):
        warnings.warn('Y does not contain any positive examples. Need Y[i] = +1 for at least 1 i.')
    if all(Y == -1):
        warnings.warn('Y does not contain any negative examples. Need Y[i] = -1 for at least 1 i.')

    if 'sample_weights' in data:
        sample_weights = data['sample_weights']
        type(sample_weights) is np.ndarray
        assert len(sample_weights) == N, 'sample_weights should contain N elements'
        assert all(sample_weights > 0.0), 'sample_weights[i] > 0 for all i '

        # by default, we set sample_weights as an N x 1 array of ones. if not, then sample weights is non-trivial
        if any(sample_weights != 1.0) and len(np.unique(sample_weights)) < 2:
            warnings.warn('note: sample_weights only has <2 unique values')

    return True

def load_data_from_csv(dataset_csv_file, sample_weights_csv_file = None, fold_csv_file = None, fold_num = 0):

    dataset_csv_file = Path(dataset_csv_file)
    if not dataset_csv_file.exists():
        raise IOError('could not find dataset_csv_file: %s' % dataset_csv_file)

    df = pd.read_csv(dataset_csv_file, sep = ',')

    raw_data = df.to_numpy()
    data_headers = list(df.columns.values)
    N = raw_data.shape[0]

    # setup Y vector and Y_name
    Y_col_idx = [0]
    Y = raw_data[:, Y_col_idx]
    Y_name = data_headers[Y_col_idx[0]]
    Y[Y == 0] = -1

    # setup X and X_names
    X_col_idx = [j for j in range(raw_data.shape[1]) if j not in Y_col_idx]
    X = raw_data[:, X_col_idx]
    variable_names = [data_headers[j] for j in X_col_idx]

    # insert a column of ones to X for the intercept
    X = np.insert(arr=X, obj=0, values=np.ones(N), axis=1)
    variable_names.insert(0, '(Intercept)')


    if sample_weights_csv_file is None:
        sample_weights = np.ones(N)
    else:
        sample_weights_csv_file = Path(sample_weights_csv_file)
        if not sample_weights_csv_file.exists():
            raise IOError('could not find sample_weights_csv_file: %s' % sample_weights_csv_file)
        sample_weights = pd.read_csv(sample_weights_csv_file, sep=',', header=None)
        sample_weights = sample_weights.to_numpy()

    data = {
        'X': X,
        'Y': Y,
        'variable_names': variable_names,
        'outcome_name': Y_name,
        'sample_weights': sample_weights,
        }

    #load folds
    if fold_csv_file is not None:
        fold_csv_file = Path(fold_csv_file)
        if not fold_csv_file.exists():
            raise IOError('could not find fold_csv_file: %s' % fold_csv_file)

        fold_idx = pd.read_csv(fold_csv_file, sep=',', header=None)
        fold_idx = fold_idx.values.flatten()
        K = max(fold_idx)
        all_fold_nums = np.sort(np.unique(fold_idx))
        assert len(fold_idx) == N, "dimension mismatch: read %r fold indices (expected N = %r)" % (len(fold_idx), N)
        assert np.all(all_fold_nums == np.arange(1, K+1)), "folds should contain indices between 1 to %r" % K
        assert fold_num in np.arange(0, K+1), "fold_num should either be 0 or an integer between 1 to %r" % K
        if fold_num >= 1:
            #test_idx = fold_num == fold_idx
            train_idx = fold_num != fold_idx
            data['X'] = data['X'][train_idx,]
            data['Y'] = data['Y'][train_idx]
            data['sample_weights'] = data['sample_weights'][train_idx]

    assert check_data(data)
    return data


def run_experiments(my_path):
    # directory of files
    files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f))]

    #res = pd.DataFrame(columns = column_names)
    res_file = os.path.join(my_path,"results_ncd_fr_YY.csv")
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
        data = load_data_from_csv(dataset_csv_file = os.path.join(my_path,f),
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
        s2 = time.time()
        coef_lr = mtds.LR_coef(data_train, weights_train.flatten())
        t2 = time.time()
        res += record_measures(data_test, f, n, p, "LR", 0, coef_lr, 1.0, t2-s2)

        # rounded logistic regression
        s3 = time.time()
        coef_rd = mtds.round_coef(coef_lr)
        t3 = time.time()
        res += record_measures(data_test, f, n, p, "Round", 0, coef_rd, 1.0, t2-s2+t3-s3)
            
        # coordinate descent from LR
        coef_lr = mtds.LR_coef(data_train, weights_train.flatten())
        s5 = time.time()
        alpha = max(np.abs(coef_lr[1:]))/10.0 # bring within range
        coef_ncd = mtds.round_coef(coef_lr, alpha)
        alpha_ncd, coef_ncd = ncd.coord_desc_nll(data_train, 1.0/alpha, coef_ncd)
        t5 = time.time()
        res += record_measures(data_test, f, n, p, "NLLCD", 0, coef_ncd, alpha_ncd, t5-s5+t2-s2) # data_test

        res_f = open(res_file, "a")
        res_f.write(res)
        res_f.close()


        i += 1
        



#if __name__ == "__main__":
#    run_experiments(sys.argv[1])



run_experiments('/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_newalg/')