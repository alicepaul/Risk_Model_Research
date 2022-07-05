
#----------- Test run ---------------------

# file_list = ['breastcancer', 'spam', 'bank', 'tbrisk_cpa','simulate5_1_data', 'simulate5_2_data', 'simulate5_3_data', 
#              'simulate5_4_data', 'simulate5_5_data', 'simulate5_6_data', 'simulate5_7_data', 'simulate5_8_data', 
#              'simulate5_9_data', 'simulate5_10_data', 'simulate10_1_data', 'simulate10_2_data','simulate10_3_data', 
#              'simulate10_4_data', 'simulate10_5_data', 'simulate10_6_data', 'simulate10_7_data', 'simulate10_8_data', 
#              'simulate10_9_data', 'simulate10_10_data', 'simulate50_1_data', 'simulate50_2_data', 'simulate50_3_data', 'simulate50_4_data', 'simulate50_5_data', 'simulate50_6_data', 'simulate50_7_data', 'simulate50_8_data', 'simulate50_9_data', 'simulate50_10_data']

# def LR_process(file_name):
#     lr_coef_df = pd.DataFrame()
#     coef_round_df = pd.DataFrame()
    
#     for i in file_list:
    
#         X, Y, variable_names = LR_read_data(i)
#         lr_coef, coef_round, lr_int, y_pred_prob, auc, accuracy, sensitivity, specificity = LR_func(X, Y)
        
#         lr_coef_df.insert(len(lr_coef_df.columns),"Coefs",lr_coef.transpose())
#         lr_coef_df.loc[-1] = ('Intercept', lr_int.transpose())
#         lr_coef_df.index = lr_coef_df.index + 1
#         coeflr_coef_df_lr_df.sort_index(inplace=True)
#         listOfSeries = [pd.Series(['Accuracy', accuracy], index=lr_coef_df.columns ) ,
#                         pd.Series(['Sensitivity', sensitivity], index=lr_coef_df.columns ) ,
#                         pd.Series(['Specificity', specificity], index=lr_coef_df.columns ),
#                         pd.Series(['AUC', auc], index=lr_coef_df.columns )]
#         lr_coef_df = lr_coef_df.append(listOfSeries,ignore_index=True)
        
#     return lr_coef_df, coef_round_df
       
# lr_coef_df, coef_round_df = LR_process(file_list)
# print(lr_coef_df.head())

