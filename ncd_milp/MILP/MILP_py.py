from pyscipopt import Model, quicksum
import numpy as np
import pandas as pd

# Load the data
file_path = "/Users/oscar/Documents/GitHub/Risk_Model_Research/MILP/data/"
file_name = "simulate0_10_0_4_data.csv"  # Replace with your file name
df = pd.read_csv(file_path + file_name) 

# Preprocessing the data
df = df.iloc[10:21, 0:5]
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

# Parameters
n, p = X.shape
M = 1000
Lambda0 = 0
SK_pool = range(-5 * p, 5 * p + 1)
PI = np.linspace(0, 1, 100)[1:-1]  # Exclude 0 and 1


def MILP_V2(X,y):

    model = Model("MILP_V2")

    # Add variable
    beta,s = {},{}
    for i in range(0,n+1):
        s[i] = model.addVar(lb=-10, ub=10, vtype="INTEGER", name="s(%s)"%i)
        for j in range(1,p+1):
            beta[i,j] = model.addVar(vtype="INTEGER", name="beta(%s,%s)"%(i,j))

    for i in range(0,n+1):
        model.addCons(quicksum(beta[i,j] * X[i,j] for (i,j) in X) - s[i] == 0)

    # Add variable
    z_ik = {}
    for i in range(0,n+1):
        for k in range(len(SK_pool)+1):
            z_ik[i,k] = model.addVar(vtype="BINARY", name=f"z_ik_{i}_{k}")
    
    for i in range(0,n+1):
        model.addCons(quicksum(z_ik[i,k] for k in range(len(SK_pool))) = 0 )

    for k in range(len(SK_pool)+1):
        model.addCons((s[i] - SK_pool[k]) - M * (1 - z_ik[i, k]) <= 0)
        model.addCons((s[i] - SK_pool[k]) + M * (1 - z_ik[i, k]) >= 0)
    
    # Add variable
    z_kl = {}
    for k in range(len(SK_pool)+1):
        for l in range(len(PI)+1):
            z_kl[k,l] = model.addVar(vtype="BINARY", name=f"z_kl_{k}_{l}")
    
    for k in range(0,len(SK_pool)+1):
        model.addCons(quicksum(z_kl[k,l] for l in range(len(PI)+1)) = 0)
    
    # Add variable
    p_il = {}
    for i in range(0,n+1):
        for l in range(len(PI)+1):
            p_il[i,l] = model.addVar(vtype="BINARY", name=f"p_il_{i}_{l}")
            model.addCons(quicksum(p_il[i,l] = 0))
    
    for i in range(0,n+1):
        for l in range(len(PI)+1):
            for k in range(0,len(SK_pool)+1):
                model.addCons(p_il[i,l] - z_kl[k,l] - z_ik[i,k] >= -1)
    


    # Objective function
    objective = -quicksum(y[i] * np.log(PI[l]) * p_il[i, l] + (1 - y[i]) * np.log(1 - PI[l]) * p_il[i, l] for i in range(n) for l in range(len(PI)))
    model.setObjective(objective, "minimize")

    model.data = beta,s,z_ik,z_kl,p_il
    return model


# Solve model
model = MILP_V2(X,y)
model.hideOutput() # silent mode
model.optimize()
cost = model.getObjVal()
print()
print("Miller-Tucker-Zemlin's model:")
print("Optimal value:", cost)
#model.printAttr("X")

x,u = model.data
sol = [i for (p,i) in sorted([(int(model.getVal(u[i])+.5),i) for i in range(1,n+1)])]
print(sol)
arcs = [(i,j) for (i,j) in x if model.getVal(x[i,j]) > .5]
sol = sequence(arcs)
print(sol)

