import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

def LR_coef(data, weights):
    #regression model
    lr_mod = LogisticRegression(penalty="none",solver="lbfgs",max_iter=1000,fit_intercept=False)
    lr_res = lr_mod.fit(data['X'], data['Y'].flatten(), weights)
    return(lr_res.coef_.flatten())

def round_coef(coef, alpha = 1.0):
    coef_new = coef.copy()/alpha
    coef_new = coef_new.round()
    coef_new[0] = coef[0]/alpha
    return(coef_new)

    
