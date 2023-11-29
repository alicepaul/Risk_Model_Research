from pyscipopt import Model, quicksum
import numpy as np
import pandas as pd

# Load the data
file_path = "/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data/"
file_name = "simulate0_10_0_4_data.csv"  # Replace with your file name
df = pd.read_csv(file_path + file_name) 

# Preprocessing the data
df = df.iloc[0:20, 1:5]
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Parameters
n, p = X.shape
M = 1000
Lambda0 = 0
SK_pool = list(range(-5 * p, 5 * p + 1))
PI = np.linspace(0, 1, 100)[1:-1]  # Exclude 0 and 1

def MILP_V2(X, y):
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

    # Objective Function
    objective = -quicksum(y[i] * np.log(PI[l]) * p_il[i, l] + (1 - y[i]) * np.log(1 - PI[l]) * p_il[i, l] for i in range(n) for l in range(len(PI)))
    model.setObjective(objective, "minimize")

    model.data = beta, s, z_ik, z_kl, p_il
    return model

# Create and solve the model
milp_model = MILP_V2(X, y)
milp_model.optimize()

milp_model.getStatus()

# Extract model data
beta, s, z_ik, z_kl, p_il = milp_model.data

# Extracting beta values
beta_values = {j: milp_model.getVal(beta[j]) for j in range(p)}

# Extracting s values
s_values = {i: milp_model.getVal(s[i]) for i in range(n)}

# Print the results
print("Beta values:")
for j in range(p):
    print(f"beta[{j}]: {beta_values[j]}")

print("\ns values:")
for i in range(n):
    print(f"s[{i}]: {s_values[i]}")
