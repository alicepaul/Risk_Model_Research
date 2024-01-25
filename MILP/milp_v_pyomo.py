from pyomo.environ import *
from pyomo.environ import SolverFactory
import numpy as np
import pandas as pd

# Load the data
file_path = "/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data/"
file_name = "simulate0_10_0_4_data.csv"  # Replace with your file name
df = pd.read_csv(file_path + file_name) 

# Preprocessing the data
df = df.iloc[0:50,0:3]
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Parameters
n, p = X.shape
M = 1000
SK_pool = np.linspace(-5 * p, 5 * p + 1,10 * p + 2,dtype=int)
PI = np.linspace(0, 1, 100)[1:-1]  # Exclude 0 and 1
#print(SK_pool)
#print(X)

# Define the pyomo model
model = ConcreteModel()

# Variables
model.beta = Var(range(p), within=Integers, bounds=(-5, 5))
model.s = Var(range(n), within=Integers, bounds=(-10, 10))
model.z_ik = Var(range(n), range(len(SK_pool)), within=Binary)
model.p_ik = Var(range(n), range(len(SK_pool)), within=NonNegativeReals, bounds=(0.0001, 0.9999))
model.p_k = Var(range(len(SK_pool)), within=NonNegativeReals, bounds=(0.0001, 0.9999))


# Constraints
def score_constraint_rule(model, i):
    return sum(model.beta[j] * X[i, j] for j in range(p)) == model.s[i]
model.score_constraint = Constraint(range(n), rule=score_constraint_rule)

def z_ik_constraint_rule(model, i):
    return sum(model.z_ik[i, k] for k in range(len(SK_pool))) == 1
model.z_ik_constraint = Constraint(range(n), rule=z_ik_constraint_rule)

def s_z_ik_constraint_rule_1(model, i, k):
    return model.s[i] - k - M * (1 - model.z_ik[i, k]) <= 0
model.s_z_ik_constraint_1 = Constraint(range(n), range(len(SK_pool)), rule=s_z_ik_constraint_rule_1)

def s_z_ik_constraint_rule_2(model, i, k):
    return model.s[i] - k + M * (1 - model.z_ik[i, k]) >= 0
model.s_z_ik_constraint_2 = Constraint(range(n), range(len(SK_pool)), rule=s_z_ik_constraint_rule_2)

def p_ik_constraint_rule_1(model, i, k):
    return model.p_ik[i, k] - model.p_k[k] <= M * (1 - model.z_ik[i, k])
model.p_ik_constraint_1 = Constraint(range(n), range(len(SK_pool)), rule=p_ik_constraint_rule_1)

def p_ik_constraint_rule_2(model, i, k):
    return model.p_ik[i, k] - model.p_k[k] >= -M * (1 - model.z_ik[i, k])
model.p_ik_constraint_2 = Constraint(range(n), range(len(SK_pool)), rule=p_ik_constraint_rule_2)

# Objective Function
def objective_rule(model):
    #return 0
    return -sum(y[i] * log(model.p_ik[i, k]) + (1 - y[i]) * log(1 - model.p_ik[i, k])
                for i in range(n) for k in range(len(SK_pool)))
model.objective = Objective(rule=objective_rule)


SolverFactory('bonmin',executable="/Users/oscar/Downloads/Bonmin-0.99.2-mac-osx-ix86-gcc4.0.1/bonmin").solve(model).write()
from scipy.optimize import minimize
