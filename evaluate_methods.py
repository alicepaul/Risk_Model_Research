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


def CPA_coef(data):
   
    # problem parameters
    max_coefficient = 100                                       # value of largest/smallest coefficient
    max_offset = 100*data['X'].shape[1]                         # maximum value of offset parameter(optional)
    max_L0_value = data['X'].shape[1]-1                         # max L0 value - set equal to p
    c0_value = 1e-6                                             # L0-penalty parameter such that c0_value > 0; larger values -> sparser models

    # create coefficient set and set the value of the offset parameter
    coef_set = riskslim.CoefficientSet(variable_names = data['variable_names'], lb = -max_coefficient, ub = max_coefficient, sign = 0)
    coef_set.update_intercept_bounds(X = data['X'], y = data['Y'], max_offset = max_offset)
    
    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set':coef_set,
    }

    # major settings (see riskslim_ex_02_complete for full set of options)
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        #
        # LCPA Settings
        'max_runtime': 30.0,                               # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': False,                     # print CPLEX progress on screen
        'loss_computation': 'fast',                         # how to compute the loss function ('normal','fast','lookup')
        #
        # LCPA Improvements
        'round_flag': True,                                # round continuous solutions with SeqRd
        'polish_flag': True,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': True,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        #
        # Initialization
        'initialization_flag': True,                       # use initialization procedure
        'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,
        #
        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }

    # train model using lattice_cpa
    model_info, mip_info, lcpa_info = riskslim.run_lattice_cpa(data, constraints, settings)

    # return coefficients
    return(model_info['solution'])

def LR_coef(data):
    #regression model
    lr_mod = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000,fit_intercept=False)
    lr_res = lr_mod.fit(data['X'], data['Y'].flatten())
    return(lr_res.coef_.flatten())

def round_coef(coef, alpha = 1.0):
    coef_new = coef.copy()/alpha
    coef_new = coef_new.round()
    coef_new[0] = coef[0]/alpha
    return(coef_new)

def get_metrics(data, coef, alpha = 1.0):

    # find v
    v = np.dot(alpha*data['X'], coef)
    v = v.astype(np.float128)
    v = np.clip(v, -709.78, 709.78)
    
    # get predicted probabilities
    prob_1 = 1.0 + np.exp(v)
    prob = np.exp(v)/(prob_1)
    pred_vals = np.zeros(shape=np.shape(prob))

    # get predicted values (0.5 cutoff)
    for i in range(len(prob)):
        if prob[i] >= 0.5:
            pred_vals[i] = 1
        else:
            pred_vals[i] = -1

    # AUC
    auc = roc_auc_score(data['Y'], prob)

    # Confusion matrix measures
    tn, fp, fn, tp = confusion_matrix(data['Y'], pred_vals, labels=[-1,1]).ravel()

    accuracy=(tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
            
    return auc, accuracy, sensitivity, specificity
    
def record_measures(f, n, p, method, measures, non_zer, res):
    data = [[f, n, p, method, measures[1], measures[2], measures[3], measures[0], non_zer]]
    new_row = pd.DataFrame(data=data, columns=column_names)
    return(pd.concat([res, new_row], ignore_index=True))


# directory of files 
my_path = "/Users/zhaotongtong/Desktop/Risk_Model_Research/sim_data"
files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f))]

# results dataframe and excel sheets
column_names = ["data", "n", "p", "method", "acc", "sens", "spec", "auc", "non-zeros"]
res = pd.DataFrame(columns = column_names)
writer = pd.ExcelWriter(os.path.join(my_path,"res_coef.xlsx"), mode='a', if_sheet_exists='replace')

# iterate through files
for f in files:

    # only do data files
    if "_data.csv" not in f:
        continue 

    # read data and add weight file if applicable
    print(f)
    sample_f =  os.path.join(my_path, f[0:(len(f)-8)]+"weights.csv")
    if (os.path.isfile(sample_f) == False):
        sample_f = None
    data = riskslim.load_data_from_csv(dataset_csv_file = os.path.join(my_path,f),
                                   sample_weights_csv_file = sample_f)
    n = data["X"].shape[0]  #rows
    p = data["X"].shape[1]-1
    
    # coefficient frame
    coef_empty = np.zeros(shape=(p+1,5))
    coef_df = pd.DataFrame(data=coef_empty, columns = ["Vars", "CPA","LR","Round","Round_Med"])
    coef_df["Vars"] = data["variable_names"]

    # cpa 
    coef_cpa = CPA_coef(data)
    cpa_measures = get_metrics(data,coef_cpa)
    res = record_measures(f,n,p,"CPA",cpa_measures,np.count_nonzero(coef_cpa), res)
    coef_df["CPA"] = coef_cpa
    
    # logistic regression
    coef_lr = LR_coef(data)
    lr_measures = get_metrics(data,coef_lr)
    res = record_measures(f,n,p,"LR",lr_measures,np.count_nonzero(coef_lr), res)
    coef_df["LR"] = coef_lr

    # rounded logistic regression
    coef_round = round_coef(coef_lr)
    lr_round_measures = get_metrics(data,coef_round)
    res = record_measures(f,n,p,"Round", lr_round_measures, np.count_nonzero(coef_round), res)
    coef_df["Round"] = coef_round

    # rounded with median
    alpha = median(coef_lr[1:len(coef_lr)])
    coef_med = round_coef(coef_lr, alpha)
    lr_med_measures = get_metrics(data,coef_med, alpha)
    res = record_measures(f,n,p,"Round_Med",lr_med_measures, np.count_nonzero(coef_med), res)
    coef_df["Round_Med"] = coef_med
    
    # write coefficient info
    coef_df.to_excel(writer, sheet_name=f, index=False)
    
    
res.to_csv(os.path.join(my_path,"results.csv"), index=False)
writer.save()
writer.close()

