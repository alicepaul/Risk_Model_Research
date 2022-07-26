import pandas as pd
import numpy as np
import os
from sklearn import datasets
from eval_f_only import *

#result by glm in R
#(Intercept)     0.8021            
# x             -4.2086      -3.0370  
# True coef: -4,-3 True intercepr: 1
X_test = np.array([[-0.6264538, 1.51178117], [0.1836433, 0.38984324], [-0.8356286, -0.62124058], [1.5952808, -2.21469989], [0.3295078, 1.12493092], [-0.8204684, -0.04493361], [0.4874291, -0.01619026], [0.7383247, 0.94383621], [0.5757814, 0.82122120], [-0.3053884, 0.59390132]])
y_test = np.array([[1],[0],[1],[1],[0],[1],[0],[0],[0],[0]])


# alpha=median or 1, beta=coef_vector, j=coef_index, gamma set to 1, lambda=0.1
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


def loss_f(n, alpha, beta, X, y, lamb_da):
    
    pd_1 = np.dot(X,beta)
    pd_1 = np.dot(y.flatten(),pd_1) * alpha
    Pr = 1.0 + np.exp(np.dot(X,beta)*alpha)
    Pr = np.sum(np.log(Pr))                               
    
    min_j = (-1/n)*np.asarray([np.sum(pd_1-Pr)]) + lamb_da* np.asarray([np.sum(abs(beta))])
    
    return min_j


def bisec_search(der_f, loss_f, a, b, beta, NMAX, TOL=1.0):
    
    if der_f(a)*der_f(b) >= 0:
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
            minimize_a = loss_f(n, alpha, beta_a, X, y, lamb_da)
            minimize_b = loss_f(n, alpha, beta_b, X, y, lamb_da)
            if minimize_a < minimize_b:
                return a 
            else: 
                return b
        
        N += 1
        if np.sign(der_f(c)) == np.sign(der_f(a)):
            a = c
        else:
            b = c
            
    return c

lr_mod = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000,fit_intercept=True)
lr_res = lr_mod.fit(X_test, y_test.flatten())
beta = lr_res.coef_.flatten()
db_j = par_deriv(10, 1, beta, X_test, y_test, 0.1, 1)
min_j = loss_f(10, 1, beta, X_test, y_test, 0.1)
print(min_j)