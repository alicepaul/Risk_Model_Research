import numpy as np
import pandas as pd
#import riskslim
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
from statistics import median

def par_deriv_nll(alpha, beta, X, y, weights, j):
    # Calculates partial derivative for beta_j in logistic regression with
    # l0 is L0 penalization, alpha*beta are current coefficients
    # Partial derivative only on NLL
    
    n = X.shape[0]
    
    # Partial derivate for loss
    warnings.filterwarnings('ignore')
    # weights = data_weight
    pd_1 = alpha*np.dot(weights, np.multiply(y, X[:,j]))
    pr = np.exp(alpha*np.dot(X,beta))
    #pr = np.clip(pr, -709.78, 709.78)
    pr = pr/(1.0+pr)
    pd_2 = alpha*(np.dot(weights, np.multiply(X[:,j], pr)))
    
    db_j_nll = (-1.0/n)*(pd_1-pd_2)
    
    
    #print("partial derivative:", db_j_nll)
    return db_j_nll


def obj_f_nll(alpha, beta, X, y, weights, l0):
    # Calculates current objective function value in logistic regression with
    # l1*L1 loss penalty and alpha*beta are current coefficients
    n = X.shape[0]
    
    # Calculate probs
    v = alpha * np.dot(X, beta)
    # print("v shape", np.shape(v))
    # print("y shape", np.shape(y))
    # print("X shape", np.shape(X))
    # print("beta shape", np.shape(beta))
    obj_1 = np.dot(weights, np.multiply(y, v))
    obj_2 = np.dot(weights, np.log(1+np.exp(v)))
    minimize_j_nll = (-1.0/n) * (obj_1-obj_2) + l0 * np.sum((beta!=0))
    
    return minimize_j_nll
    

def bisec_search(alpha, beta, X, y, weights, l0, j, a=-10, b=10, TOL=1.0):
    # Runs bisection search on beta_j to find the best value

    # Starting beta values and derivatives
    beta_a = beta.copy()
    beta_a[j] = a
    beta_b = beta.copy()
    beta_b[j] = b
    #NLL(0)
    beta_0 = np.zeros(np.shape(beta_a))
    der_f_a = par_deriv_nll(alpha, beta_a, X, y, weights, j)
    der_f_b = par_deriv_nll(alpha, beta_b, X, y, weights, j)
    
    # Check that 0 derivative in range
    search = True
    if der_f_a > 0 or der_f_b < 0:
        search = False
    
    # Bisection search
    while b-a > 1 and search:
        # Find mid point
        c = math.floor((a+b)/2)
        beta_c = beta.copy()
        beta_c[j] = c
        der_f_c = par_deriv_nll(alpha, beta_c, X, y, weights, j)
        if der_f_c == 0:
            #print("Found solution:", c)
            return(beta_c)
        
        # Check where to recurse
        if np.sign(der_f_c) == np.sign(der_f_a):
            a = c
            beta_a = beta_c
            der_f_a = der_f_c
        else:
            b = c
            beta_b = beta_c
            der_f_b = der_f_c
    
    # Find best of b and a in objective function
    obj_a = obj_f_nll(alpha, beta_a, X, y, weights, l0)
    obj_b = obj_f_nll(alpha, beta_b, X, y, weights, l0)
    ### NEW : comapre NLL(b_j)+l0 < NLL(0)?
    obj_0 = obj_f_nll(alpha, beta_0, X, y, weights, l0)
    if obj_a < obj_b and obj_a < obj_0:
        return beta_a
    #should be else????
    elif obj_0 < obj_a and obj_0 < obj_b:
        return beta_0
    return beta_b
    


def update_alpha(beta, X, y, weights):
    # Run logistic regression on current integer scores

   
    # Calculate scores - ignores intercept
    zi = np.dot(X[:,1:], beta[1:])

    # Runs logistic regression and finds alpha and beta_0
    lr_mod = LogisticRegression(penalty="none")
    ###?? y_train
    lr_res = lr_mod.fit(np.reshape(zi, (-1,1)), y, weights)
    new_coef = lr_res.coef_.flatten()
    alpha = new_coef[0]
    beta[0] = lr_res.intercept_/alpha
    
    return alpha, beta
    

def coord_desc_nll(data, alpha, beta, l0 = 0.0, max_iter = 100, tol= 1e-5):
    # Runs coordinate descent on algorithm from starting point alpha beta
    
   
    X = data['X']
    n = X.shape[0]
    weights = data['sample_weights'].flatten()
    

    ytemp = data['Y']
    y = np.zeros(shape=(n))
    for i in range(n):
        if ytemp[i] == 1:
            y[i] = 1
    
    p = X.shape[1]
    iters = 0
    while iters < max_iter:
        old_beta = beta.copy()
        # Coodinate descent for each j
        for j in range(1,p):
            beta = bisec_search(alpha, beta, X, y, weights, l0, j)
            alpha, beta = update_alpha(beta, X, y, weights)

        # Check if change in beta is within tolerance to converge
        if max(np.abs(old_beta - beta)) < tol:
            break
        iters += 1
    
    return(alpha, beta)
    
