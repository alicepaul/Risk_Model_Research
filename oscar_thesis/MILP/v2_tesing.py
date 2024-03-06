from pyscipopt import Model, quicksum
import milpv2 as milp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from statistics import median

# Load the data
file_path = "/Users/oscar/Documents/GitHub/Risk_Model_Research/ncd_milp/sim_milp/"
file_name = "sim_50_3_2_7_1_1_data.csv"  # Replace with your file name
df = pd.read_csv(file_path + file_name) 

# Preprocessing the data
#df = df.iloc[0:20, 0:4]
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values


# Parameters
n, p = X.shape
M = 1000
Lambda0 = 0
SK_pool = list(range(-5 * p, 5 * p + 1))
PI = np.linspace(0, 1, 100)[1:-1]  # Exclude 0 and 1


# Create and solve the model
milp_model = milp.MILP_V2(X,y,n,p,M,SK_pool,PI)
milp_model.optimize()



    # Step 2: Check if the problem is solved to optimality or within 1% gap
    status = milp_model.getStatus()
    gap = milp_model.getGap()

    if status == 'optimal' or gap <= 0.01:
        print("Solution is optimal or within 1% gap.")
        model.data = beta, s, z_ik, z_kl, p_il
        return model
    else:
    # Step 3: Warm start with the solution from the last run
    # Get the current best solution
        current_solution = milp_model.getBestSol()

    # Reset the model if you want to change parameters or re-solve from scratch
        milp_model.freeTransform()

    # Optionally adjust the model (e.g., change parameters)
        milp_model.setParam('limits/time', 200)  # Set a new time limit for the next run

    # Add the previous solution as a starting point
        milp_model.addSol(current_solution)

    # Re-optimize the model
        milp_model.optimize()

    # Check the new solution status and gap
        new_status = milp_model.getStatus()
        new_gap = milp_model.getGap()
        print(f"New status: {new_status}, New gap: {new_gap}")

# Extract model data
beta, s, z_ik, z_kl, p_il = milp_model.data

# Extracting Solving time
solving_time = milp_model.getSolvingTime()

# Extracting beta values
beta_values = {j: milp_model.getVal(beta[j]) for j in range(p)}
#s_scores = [row[0] * beta_values[0] + row[1] * beta_values[1] for row in X]

# Change values
X = np.array(X)
beta_values = np.array(list(beta_values.values()))
np.dot(X, beta_values)
# Extracting s values
s_values = {i: milp_model.getVal(s[i]) for i in range(n)}

# Extracting zkl values
zkl_values = {(k, l): milp_model.getVal(z_kl[k, l]) for k in SK_pool for l in range(len(PI))}

for k in SK_pool:
    for l in range(len(PI)):
       print(f"z_kl[{k},{l}]: {zkl_values[k, l]}")

# Extracting keys where z_kl values are equal (or almost equal) to 1
keys_zkl = {key: val for key, val in zkl_values.items() if val==1}
new_dict = {key[0]: key[1] for key in keys_zkl.keys()}


probs = []
for val in list(s_values.values()):
    if val in new_dict:
        score_value = new_dict[val]
        probs.append(PI[score_value])
    else:
        probs.append(None)


def get_metrics(y, probs):

    # Ensure probs is a list of probabilities
    if isinstance(probs, dict):
        probs = [probs[i] for i in range(len(probs))]

    # Ensure probs is a 1D array
    probs = np.array(probs).ravel()

    # Transform predicted probabilities into binary classes
    predicted_class = [1 if prob > 0.5 else 0 for prob in probs]

    # AUC
    auc = roc_auc_score(y, probs)

    # Confusion matrix measures
    tn, fp, fn, tp = confusion_matrix(y, predicted_class, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return auc, accuracy, sensitivity, specificity




get_metrics(y,probs)
