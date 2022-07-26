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

X_test = np.array([[1, 1], [1, 2], [1,3], [1,4], [1, 5], [1, 6]])
y_test = np.array([[0],[0],[0],[0],[1],[1]])


def LR_coef(X, y):
    #regression model
    lr_mod = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000,fit_intercept=False)
    lr_res = lr_mod.fit(X, y.flatten())
    return(lr_res.coef_.flatten())


# alpha=median or 1, beta=coef_vector, j=coef_index, gamma set to 1, lambda=0.1
def par_deriv(n, alpha, beta, X, y,lamb_da, j):
    
    # TODO: ATTENTION! swtich X[:,j] and y???
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

print(np.shape(X_test))
print(np.shape(y_test))
coef_lr = LR_coef(X_test,y_test)
beta = coef_lr
b_0 = coef_lr[0]

print(beta)

print(X_test[1])
# db_j = par_deriv(3, 1, beta, X_test, y_test,0.1, 1)

# print(db_j)
# f = 
# c = bisec_search(f,-10, 10, 5, TOL=1.0)
# print(c)    
    


