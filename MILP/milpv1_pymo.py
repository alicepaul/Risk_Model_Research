from pyomo.environ import *
from pyomo.environ import SolverFactory
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
SK_pool = range(-5 * p, 5 * p + 1)
PI = np.linspace(0, 1, 100)[1:-1]  # Exclude 0 and 1

# Define the pyomo model
model = ConcreteModel()

# Variables
model.beta = Var(range(p), within=Integers, bounds=(-5, 5))
model.s = Var(range(n), within=Integers, bounds=(-10, 10))
model.z_ik = Var(range(n), SK_pool, within=Binary)
model.p_ik = Var(range(n), SK_pool, within=Reals, bounds=(0.0001, 0.9999))

# Constraints
def score_constraint_rule(model, i):
    return sum(model.beta[j] * X[i, j] for j in range(p)) == model.s[i]
model.score_constraint = Constraint(range(n), rule=score_constraint_rule)

# Add other constraints following the same pattern

# Objective Function
def objective_rule(model):
    return -sum(y[i] * log(model.p_ik[i, k]) + (1 - y[i]) * log(1 - model.p_ik[i, k]) 
                for i in range(n) for k in SK_pool)
model.objective = Objective(rule=objective_rule, sense=minimize)


# Choose a suitable solver available in your environment
solver = SolverFactory('cbc')
results = solver.solve(model, tee=True)


beta_values = [value(model.beta[j]) for j in range(p)]
s_values = [value(model.s[i]) for i in range(n)]

print("Beta values:", beta_values)
print("s values:", s_values)
