from pyscipopt import Model, quicksum,log
import numpy as np
import pandas as pd

# Load the data
file_path = "/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data/"
file_name = "simulate0_10_0_4_data.csv"  # Replace with your file name
df = pd.read_csv(file_path + file_name) 

# Preprocessing the data
df = df.iloc[0:20,1:5]
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Parameters
n, p = X.shape
M = 1000
Lambda0 = 0
SK_pool = list(range(-5 * p, 5 * p + 1))
PI = np.linspace(0, 1, 100)[1:-1]  # Exclude 0 and 1

def MILP_V1(X,y,n,p,M,SK_pool,PI):
    model = Model("MILP_V1")

    # Variables
    beta = {j: model.addVar(vtype="INTEGER", name="beta_{}".format(j)) for j in range(p)}
    s = {i: model.addVar(vtype="INTEGER", name="s_{}".format(i)) for i in range(n)}
    z_ik = {(i, k): model.addVar(vtype="BINARY", name="z_ik_{}_{}".format(i, k)) for i in range(n) for k in SK_pool}
    p_ik = {(i, k): model.addVar(lb=0.0001, ub=0.9999, vtype="CONTINUOUS", name="p_ik_{}_{}".format(i, k))
        for i in range(n) for k in SK_pool}

    # Constraints
    for i in range(n):
        model.addCons(quicksum(beta[j] * X[i, j] for j in range(p)) == s[i])
        model.addCons(quicksum(z_ik[i, k] for k in SK_pool) == 1)

        for k in SK_pool:
            model.addCons(s[i] - k - M * (1 - z_ik[i, k]) <= 0)
            model.addCons(s[i] - k + M * (1 - z_ik[i, k]) >= 0)

            for l in range(len(PI)):
                model.addCons(p_ik[i, k] - PI[l] <= M * (1 - z_ik[i, k]))
                model.addCons(p_ik[i, k] - PI[l] >= -M * (1 - z_ik[i, k]))



    # Objective Function
    objective = -quicksum(y[i] * log(p_ik[i,k])  + (1 - y[i]) * log(1 - p_ik[i,k]) for i in range(n) for k in SK_pool )
    model.setObjective(objective, "minimize")

    model.data = beta, s, z_ik, p_ik
    return model


# Create and solve the model
milp_model_v1 = MILP_V1(X,y,n,p,M,SK_pool,PI)
milp_model_v1.optimize()

milp_model_v1.getStatus()

# Extract model data
beta, s, z_ik, p_ik = milp_model_v1.data

# Extracting beta values
beta_values_v1 = {j: milp_model_v1.getVal(beta[j]) for j in range(p)}

# Extracting s values
s_values_v1 = {i: milp_model_v1.getVal(s[i]) for i in range(n)}

# Print the results
print("Beta values:")
for j in range(p):
    print(f"beta[{j}]: {beta_values_v1[j]}")

print("\ns values:")
for i in range(n):
    print(f"s[{i}]: {s_values_v1[i]}")

