import milpv2 as milp
import os
import pandas as pd
from pyscipopt import Model, quicksum
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from statistics import median
from sklearn.linear_model import LogisticRegression


column_names = "data, n, p, method, auc, acc, sens, spec, non-zeros, time \n"

def get_metrics(X_test,y_test,beta_values,score_board,PI):

    # Convert test X into S 
    X_test = np.array(X_test)
    beta_values = np.array(list(beta_values.values()))
    s_val = np.dot(X_test, beta_values)


    # convert s into probs
    probs = []
    for val in s_val:
        if val in score_board:
            score_value = score_board[val]
            probs.append(PI[score_value])
        else:
            probs.append(None)

    # Ensure probs is a 1D array
    probs = np.array(probs).ravel()

    # Transform predicted probabilities into binary classes
    predicted_class = [1 if prob > 0.5 else 0 for prob in probs]

    # AUC
    auc = roc_auc_score(y_test, probs)

    # Confusion matrix measures
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_class, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return auc, accuracy, sensitivity, specificity

def record_measures(X_test,y_test,beta_values,score_board,filename,n,p,method,time,PI):
    # Adds record
    measures = get_metrics(X_test,y_test,beta_values,score_board,PI)

    res_str = filename + "," + str(n) + "," + str(p) + "," + method + "," + str(measures[0]) + "," + str(measures[1]) + "," \
    + str(measures[2]) + "," + str(measures[3]) + "," + str(np.count_nonzero(beta_values)) + ","+str(time) +"\n"
    
    return(res_str)

def process_files_and_predict(working_directory):

    res_file = os.path.join(working_directory,"results_milp_v2_testing_2.20_t_limit.csv")
    res_f = open(res_file, "w")
    res_f.write(column_names)
    res_f.close()

    for filename in os.listdir(working_directory):
        if filename.endswith('_data.csv'):
            res = ''
            # Load and preprocess data
            df = pd.read_csv(os.path.join(working_directory, filename))
            #df = df.iloc[0:20, 0:3]  # Adjust slicing based on data structure
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values

            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

            # Define parameters (adjust these as needed)
            n, p = X_train.shape
            M = 1000 # some big number to control for constraints 
            SK_pool = list(range(-5 * p, 5 * p + 1))
            PI = np.linspace(0, 1, 100)[1:-1]

            # Apply MILP_V2 model (implementation should be provided)
            milp_model = milp.MILP_V2(X_train,y_train,n,p,M,SK_pool,PI)
            milp_model.optimize()
            milp_model.getStatus()
            
            # Extract model data
            beta, s, z_ik, z_kl, p_il = milp_model.data

            # Extracting Solving time
            solving_time = milp_model.getSolvingTime()

            # Extracting beta values
            beta_values = {j: milp_model.getVal(beta[j]) for j in range(p)}

            # Extracting s values
            s_values = {i: milp_model.getVal(s[i]) for i in range(n)}

            # Extracting zkl values
            zkl_values = {(k, l): milp_model.getVal(z_kl[k, l]) for k in SK_pool for l in range(len(PI))}

            # Extracting keys where z_kl values are equal (or almost equal) to 1
            keys_zkl = {key: val for key, val in zkl_values.items() if val==1}

            # Form score to prob board: score to an l (index of the probability vector, PI)
            score_board  = {key[0]: key[1] for key in keys_zkl.keys()}

            # Store results
            res += record_measures(X_test,y_test,beta_values,score_board,filename,n,p,'v2',solving_time,PI)

            res_f = open(res_file, "a")
            res_f.write(res)
            res_f.close()

# Usage example
working_directory = '/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_milp'  # Replace with your actual directory path
process_files_and_predict(working_directory)

