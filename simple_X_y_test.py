import pandas as pd
import numpy as np
import os
from sklearn import datasets
from eval_f_only import *

#result by glm in R
#(Intercept)    1.927      
# x             -5.150
# True coef: -4, True intercepr: 1
X_test = np.array([[-0.6264538], [0.1836433], [-0.8356286], [1.5952808], [0.3295078], [-0.8204684], [0.4874291], [0.7383247], [0.5757814], [-0.3053884]])
y_test = np.array([[1],[1],[1],[0],[0],[1],[0],[0],[1],[1]])



# alpha=median or 1, beta=coef_vector, j=coef_index, gamma set to 1, lambda=0.1
def par_deriv(n, alpha, beta, X, y,lamb_da, j):
    
    pd_1 = np.dot(X[:,j],y) * beta[j] * alpha
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
    pd_1 = np.dot(y,pd_1) * alpha
    Pr = 1.0 + np.exp(np.dot(X,beta)*alpha)
    Pr = np.sum(np.log(Pr))                               
    
    min_j = (-1.0/n)*[np.sum(pd_1-Pr)] + lamb_da*np.sum(abs(beta))
    
    return min_j


def bisec_search(der_f, loss_f, a, b, NMAX, TOL=1.0):
    
    if der_f(a)*der_f(b) >= 0:
        print("Bisection's condition is not satisfied")
        return None
    
    N = 1
    
    while N <= NMAX:
        # TODO: put floor make sure c is integer
        c = math.floor((a+b)/2)
        if der_f(c) == 0:
            print("Found solution:", c)
            break
        
        # # TODO: check the actual LR loss function to make sure a or b
        # # call loss_f here to check which is best
        N += 1
        if np.sign(der_f(c)) == np.sign(der_f(a)):
            a = c
        else:
            b = c
    
        # loss_f(n, alpha, beta, X, y, lambda)
            
    return c 


lr_mod = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000,fit_intercept=True)
lr_res = lr_mod.fit(X_test, y_test.flatten())
coef_lr = lr_res.coef_.flatten()

beta = coef_lr
b_0 = lr_res.intercept_

print("coef",beta)
print("intercept",b_0)


