import numpy as np
import pandas as pd
import riskslim
import math
from sklearn.linear_model import LogisticRegression
import warnings
from statistics import median

def par_deriv(alpha, beta, X, y, weights, l1, j):
    # Calculates partial derivate for beta_j in logistic regression with
    # l1*L1 loss penalty and alpha*beta are current coefficients
    
    n = X.shape[0]
    
    # Partial derivate for loss
    warnings.filterwarnings('ignore') # TODO: remove later
    pd_1 = alpha*np.dot(weights, np.multiply(y, X[:,j]))
    pr = np.exp(alpha*np.dot(X,beta))
    #pr = np.clip(pr, -709.78, 709.78)
    pr = pr/(1.0+pr)
    pd_2 = alpha*(np.dot(weights, np.multiply(X[:,j], pr)))
    
    db_j = (-1.0/n)*(pd_1-pd_2)
    
    # Add penalty term
    penalty = 0
    if beta[j] > 0:
        penalty = l1
    elif beta[j] < 0:
        penalty = -l1

    #print("partial derivative:", db_j)
    return db_j+penalty

def obj_f(alpha, beta, X, y, weights, l1):
    # Calculates current objective function value in logistic regression with
    # l1*L1 loss penalty and alpha*beta are current coefficients
    n = X.shape[0]
    
    # Calculate probs
    v = alpha * np.dot(X, beta)
    
    obj_1 = np.dot(weights, np.multiply(y, v))
    obj_2 = np.dot(weights, np.log(1+np.exp(v)))
    minimize_j = (-1.0/n) * (obj_1-obj_2) + l1 * np.sum(np.abs(beta))
    
    return minimize_j
    
#def full_search(alpha, beta, X, y, l1, j, a=-10, b=10, TOL=1.0):
#    obj = obj_f(alpha, beta, X, y, l1)
#    for c in range(a, b+1):
#        beta_c = beta.copy()
#        beta_c[j] = c
#        obj_c = obj_f(alpha, beta_c, X, y, l1)
#        if obj_c < obj:
#            obj = obj_c
#            beta = beta_c
#    return(beta)

def bisec_search(alpha, beta, X, y, weights, l1, j, a=-10, b=10, TOL=1.0):
    # Runs bisection search on beta_j to find the best value

    # Starting beta values and derivatives
    beta_a = beta.copy()
    beta_a[j] = a
    beta_b = beta.copy()
    beta_b[j] = b
    der_f_a = par_deriv(alpha, beta_a, X, y, weights, l1, j)
    der_f_b = par_deriv(alpha, beta_b, X, y, weights, l1, j)
    
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
        der_f_c = par_deriv(alpha, beta_c, X, y, weights, l1, j)
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
    obj_a = obj_f(alpha, beta_a, X, y, weights, l1)
    obj_b = obj_f(alpha, beta_b, X, y, weights, l1)
    if obj_a < obj_b:
        return beta_a
    return beta_b


def update_alpha(beta, X, y, weights):
    # Run logistic regression on current integer scores

    # Calculate scores - ignores intercept
    zi = np.dot(X[:,1:], beta[1:])

    # Runs logistic regression and finds alpha and beta_0
    lr_mod = LogisticRegression(penalty="none")
    lr_res = lr_mod.fit(np.reshape(zi, (-1,1)), y, weights)
    new_coef = lr_res.coef_.flatten()
    alpha = new_coef[0]
    beta[0] = lr_res.intercept_/alpha
    
    return alpha, beta
    

def coord_desc(data, alpha, beta, l1 = 0.0, max_iter = 100, tol= 1e-5):
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
            beta = bisec_search(alpha, beta, X, y, weights, l1, j)
            alpha, beta = update_alpha(beta, X, y, weights)

        # Check if change in beta is within tolerance to converge
        if max(np.abs(old_beta - beta)) < tol:
            break
        iters += 1
    
    return(alpha, beta)
    
