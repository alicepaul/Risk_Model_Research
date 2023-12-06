from pyscipopt import Model, quicksum
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from statistics import median
from sklearn.linear_model import LogisticRegression

def MILP_V2(X,y,n,p,M,SK_pool,PI):
    model = Model("MILP_V2")

    # Variables
    beta = {j: model.addVar(vtype="INTEGER", name="beta_{}".format(j)) for j in range(p)}
    s = {i: model.addVar(vtype="INTEGER", name="s_{}".format(i)) for i in range(n)}
    z_ik = {(i, k): model.addVar(vtype="BINARY", name="z_ik_{}_{}".format(i, k)) for i in range(n) for k in SK_pool}
    z_kl = {(k, l): model.addVar(vtype="BINARY", name="z_kl_{}_{}".format(k, l)) for k in SK_pool for l in range(len(PI))}
    p_il = {(i, l): model.addVar(vtype="BINARY", name="p_il_{}_{}".format(i, l)) for i in range(n) for l in range(len(PI))}

    # Constraints
    for i in range(n):
        model.addCons(quicksum(beta[j] * X[i, j] for j in range(p)) == s[i])
        model.addCons(quicksum(z_ik[i, k] for k in SK_pool) == 1)
        model.addCons(quicksum(p_il[i, l] for l in range(len(PI))) == 1)

        for k in SK_pool:
            model.addCons(s[i] - k - M * (1 - z_ik[i, k]) <= 0)
            model.addCons(s[i] - k + M * (1 - z_ik[i, k]) >= 0)

            for l in range(len(PI)):
                model.addCons(p_il[i, l] - z_kl[k, l] - z_ik[i, k] >= -1)

    for k in SK_pool:
        model.addCons(quicksum(z_kl[k, l] for l in range(len(PI))) == 1)

    # constraints for ensuring higher k is associated with higher probability PI[l]
    #for k in SK_pool[1:]:  # Starting from the second element in SK_pool
    #    for l in range(len(PI) - 1):  # l < l', so we don't include the last element
    #        for l_prime in range(l + 1, len(PI)):
    #            model.addCons(z_kl[k, l] <= 1 - z_kl[k - 1, l_prime])


    # Objective Function
    objective = -quicksum(y[i] * np.log(PI[l]) * p_il[i, l] + (1 - y[i]) * np.log(1 - PI[l]) * p_il[i, l] for i in range(n) for l in range(len(PI)))
    model.setObjective(objective, "minimize")


    # Run Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    
    # Extract and round coefficients
    beta_start = np.round(log_reg.coef_[0]).astype(int)

    # Create a partial solution
    partial_solution = model.createPartialSol()

    # Set initial values for beta using beta_start
    for j, val in enumerate(beta_start):
        model.setSolVal(partial_solution, beta[j], val)

    # Calculate s values using X and beta_start
    for i in range(n):
        s_val = sum(beta_start[j] * X[i, j] for j in range(p))
        model.setSolVal(partial_solution, s[i], s_val)
    
    # Add the partial solution to the model
    model.addSol(partial_solution)

    model.data = beta, s, z_ik, z_kl, p_il
    return model

