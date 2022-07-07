import os
import pprint
import numpy as np
import riskslim
import pandas as pd
import pickle
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def CPA_function(data_name, max_L0_value):
    # data

    data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
    data_csv_file = data_dir + data_name + '_data.csv'          # csv file for the dataset
    sample_weights_csv_file = None                              # csv file of sample weights for the dataset (optional)
    
    # problem parameters
    max_coefficient = 5                                         # value of largest/smallest coefficient
    # max_L0_value = 12                                            # maximum model size (set as float(inf))
    max_offset = 50                                             # maximum value of offset parameter(optional)

    c0_value = 1e-6                                             # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms

    # load data from disk
    data = riskslim.load_data_from_csv(dataset_csv_file = data_csv_file, sample_weights_csv_file = sample_weights_csv_file)

    df = pd.read_csv(data_csv_file)

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
        'display_cplex_progress': True,                     # print CPLEX progress on screen
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

    #print model contains model
    # riskslim.print_model(model_info['solution'], data)

    #model info contains key results
    # pprint.pprint(model_info)
    
    v = np.dot(data['X'], model_info['solution'])
    v = v.astype(np.float128)
    
    # print(np.isnan(v))
    # print(np.isnan(data['Y']))
    
    np.seterr(invalid='ignore')
    # get pred vals and pred prob
    prob_1 = 1.0 + np.exp(v)
    prob = np.exp(v)/(prob_1)
    # prob = np.array(prob)
    pred_vals = np.zeros(shape=np.shape(prob))
    
    for i in range(len(prob)):
        if prob[i] >= 0.5:
            pred_vals[i] = 1
        else:
            pred_vals[i] = 0
    
    # pred_vals = np.array(pred_vals)
    
    auc = roc_auc_score(data['Y'], prob)
    print(data['Y'])
    print(pred_vals)

    #specificity, sensitivity
    cm = confusion_matrix(data['Y'], pred_vals, labels = [0,1])
    print('Confusion Matrix : \n', cm)

    total=sum(sum(cm))
    #####from confusion matrix calculate accuracy
    accuracy=(cm[0,1]+cm[2,2])/total
    # print ('Accuracy : ', accuracy)

    sensitivity = cm[0,1]/(cm[0,1]+cm[0,2])
    # print('Sensitivity : ', sensitivity)

    specificity = cm[2,2]/(cm[2,1]+cm[2,2])
    # print('Specificity : ', specificity)
            
    return model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df

    


#-------sample CPA-----------

# file_list = ['breastcancer', 'spambase', 'bank', 'tbrisk_cpa','simulate5_1_data', 'simulate5_2_data', 'simulate5_3_data', 
#              'simulate5_4_data', 'simulate5_5_data', 'simulate5_6_data', 'simulate5_7_data', 'simulate5_8_data', 
#              'simulate5_9_data', 'simulate5_10_data', 'simulate10_1_data', 'simulate10_2_data','simulate10_3_data', 
#              'simulate10_4_data', 'simulate10_5_data', 'simulate10_6_data', 'simulate10_7_data', 'simulate10_8_data', 
#              'simulate10_9_data', 'simulate10_10_data', 'simulate50_1_data', 'simulate50_2_data', 'simulate50_3_data', 'simulate50_4_data', 'simulate50_5_data', 'simulate50_6_data', 'simulate50_7_data', 'simulate50_8_data', 'simulate50_9_data', 'simulate50_10_data']
#----------------------


# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('breastcancer', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='breast.dat.cpa_') 

# #--------------------


# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('spambase', 57)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='spambase.dat.cpa_') 

# #--------------------


# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('bank', 57)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='bank.dat.cpa_') 

#--------------------



model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('tbrisk_cpa', 12)

# print(accuracy)

# Write to file

model_result = pd.DataFrame(list(df.columns)).copy()
model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
model_result.rename(columns = {0:'Features'}, inplace = True)
filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
                pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
                pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
                pd.Series(['AUC', auc], index=filter_model_result.columns )]
filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
                    mode='a') as writer:  
    filter_model_result.to_excel(writer, sheet_name='tbrisk_cpa.dat.cpa_') 





# ---------------
# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_1', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_1.dat.cpa_') 


# #---------------------------

# # ---------------
# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_2', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_2.dat.cpa_') 


# #---------------------------

# # ---------------
# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_3', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_3.dat.cpa_') 


# #---------------------------

# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_4', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_4.cpa_') 




# #----------------

# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_5', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_5.dat.cpa_') 


# # ---------------
# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_6', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_6.dat.cpa_') 


# #---------------------------

# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_7', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_7.cpa_') 




# #----------------

# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_8', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_8.dat.cpa_') 
    
# # ---------------
# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_9', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_9.dat.cpa_') 


# #---------------------------

# model_info, prob, pred_vals, auc, accuracy, sensitivity, specificity, df = CPA_function('simulate5_10', 9)

# # print(accuracy)

# # Write to file

# model_result = pd.DataFrame(list(df.columns)).copy()
# model_result.insert(len(model_result.columns),"Coefs",model_info['solution'])
# model_result.rename(columns = {0:'Features'}, inplace = True)
# # model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/cpa_breastcancer.csv')
# filter_model_result = model_result[(model_result['Coefs'] != 0.0) & (model_result['Coefs'] !=-0.0) ]
# listOfSeries = [pd.Series(['Accuracy', accuracy], index=filter_model_result.columns ) ,
#                 pd.Series(['Sensitivity', sensitivity], index=filter_model_result.columns ) ,
#                 pd.Series(['Specificity', specificity], index=filter_model_result.columns ),
#                 pd.Series(['AUC', auc], index=filter_model_result.columns )]
# filter_model_result = filter_model_result.append(listOfSeries,ignore_index=True)
# filter_model_result.to_csv('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/filter_cpa_breastcancer.csv')
# with pd.ExcelWriter('/Users/zhaotongtong/Desktop/Risk_Model_Research/risk-slim/examples/results/result_compare_update.xlsx',
#                     mode='a') as writer:  
#     filter_model_result.to_excel(writer, sheet_name='sim5_10.cpa_') 


