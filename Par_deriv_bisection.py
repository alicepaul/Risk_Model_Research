import pandas as pd
import numpy as np
import os
import riskslim
import math
from eval_f_only import *

# directory of files 
my_path = "/Users/zhaotongtong/Desktop/Risk_Model_Research/test_data"
files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f))]


# iterate through files
for f in files:

    # only do data files
    if "_data.csv" not in f:
        continue 

    # read data and add weight file if applicable
    print(f)
    sample_f =  os.path.join(my_path, f[0:(len(f)-8)]+"weights.csv")
    if (os.path.isfile(sample_f) == False):
        sample_f = None
    data = riskslim.load_data_from_csv(dataset_csv_file = os.path.join(my_path,f),
                                   sample_weights_csv_file = sample_f)
    n = data["X"].shape[0]  # number of observations
    p = data["X"].shape[1]-1 # number of features
    print("p",p)
    print("n",n)   
    X = data['X']
    y = data['Y']
    
    # coefficient frame
    coef_empty = np.zeros(shape=(p+1,4))
    coef_df = pd.DataFrame(data=coef_empty, columns = ["Vars","LR","Round","Round_Med"])
    coef_df["Vars"] = data["variable_names"]

 
    # logistic regression
    coef_lr = LR_coef(data)
    lr_measures = get_metrics(data,coef_lr)
    beta = coef_lr
    print(np.shape(beta))
    b_0 = coef_lr[0]
    print("b_0",b_0)
    
    # rounded logistic regression
    coef_round = round_coef(coef_lr)
    lr_round_measures = get_metrics(data,coef_round)
    beta_round = coef_round
    b_0_round = coef_round[0]
    print("b_0_round",b_0_round)
    
    # rounded with median
    alpha = median(coef_lr[1:len(coef_lr)])
    coef_med = round_coef(coef_lr, alpha)
    lr_med_measures = get_metrics(data,coef_med, alpha)
    beta_med = coef_med
    b_0_med = coef_med[0]
    print("b_0_med",b_0_med)
 # ----------------------------------------------------  
    #Toke out file written part


# alpha=median or 1, beta=coef_vector, j=coef_index, gamma set to 1, lamb_da=0.1
def par_deriv(n, alpha, beta, X, y,lamb_da, j):
    pd_1 = np.dot(y.flatten(), X[:,j]) * beta[j] * alpha
    Pr = np.exp(np.dot(X,beta)*alpha)
    Pr = Pr/(1.0+Pr)                               
    pd_2 = alpha*(np.dot(X[:,j], Pr))
    
    if beta[j] > 0:
        db_j = (-1/n)*np.asarray([np.sum(pd_1-pd_2)]) + lamb_da
    elif beta[j] < 0:
        db_j = (-1/n)*np.asarray([np.sum(pd_1-pd_2)]) - lamb_da
    elif beta[j] == 0: 
        db_j = (-1/n)*np.asarray([np.sum(pd_1-pd_2)])

    print("partial derivative:", db_j)
    return db_j

#loss_minimization
def loss_f(n, alpha, beta, X, y, lamb_da):
    pd_1 = np.dot(X, beta)
    pd_1 = np.dot(y.flatten(), pd_1) * alpha
    Pr = 1.0 + np.exp(np.dot(X, beta) * alpha)
    Pr = np.sum(np.log(Pr))                               
    minimize_j = (-1/n) * np.asarray([np.sum(pd_1-Pr)]) + lamb_da * np.asarray([np.sum(abs(beta))])
    
    return minimize_j

# der_f = par_deriv
def bisec_search(der_f, loss_f, a, b, beta, NMAX, TOL=1.0):
    beta_j_a = a
    beta_j_b = b
    der_f_a = par_deriv(n, alpha, beta, data['X'], data['Y'],0.1, beta_j_a)
    der_f_b = par_deriv(n, alpha, beta, data['X'], data['Y'],0.1, beta_j_b)
    if der_f_a * der_f_b >= 0:
        print("Bisection's condition is not satisfied")
        return None
    
    N = 1
    while N <= NMAX:
        c = math.floor((a+b)/2)
        if der_f(c) == 0:
            print("Found solution:", c)
            break
        
        #loss function check
        if b-a == 1:
            beta_a = beta
            beta_a[j] = a
            beta_b = beta
            beta_b[j] = b
            minimize_a = loss_f(n, alpha, beta_a[j], X, y, lamb_da)
            minimize_b = loss_f(n, alpha, beta_b[j], X, y, lamb_da)
            if minimize_a < minimize_b:
                return a 
            else: 
                return b
        
        #sign check
        N += 1
        if np.sign(der_f(c)) == np.sign(der_f(a)):
            a = c
        else:
            b = c
            
    return c


def update_b0_a(data,beta):
    coef_only= np.delete(beta, (0), axis=0)
    X_i = data['X'][:,1:]
    zi = np.dot(X_i,coef_only)
    lr_mod = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000,fit_intercept=False)
    lr_res = lr_mod.fit(zi, data['Y'].flatten())
    new_coef = lr_res.coef_.flatten()
    alpha = new_coef[1]
    b0_update = new_coef[0]
    b0_final = b0_update/alpha
    
    return b0_final, alpha
    


