import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau
from pathlib import Path
import os
import statistics
import statsmodels.stats.multitest as smt


#path of the main folder
main_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ/EGFR_prediction/test_samples/iucpq-Venkata/')


#read the features from Excel files
df_pyrads_data = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis_new/clinical_IHC_individual_inter_with_rads_bw25.xlsx'), sheet_name='pyrads-common-racat' )
#df_IHC_data = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='clinical_data' )
df_IHC_data = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis_new/clinical_IHC_individual_inter_with_rads_bw25.xlsx'), sheet_name='IHC' )
df_racat_data = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis_new/clinical_IHC_individual_inter_with_rads_bw25.xlsx'), sheet_name='racat-common-pyrads')  
df_features = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis_new/clinical_IHC_individual_inter_with_rads_bw25.xlsx'), sheet_name='common-features')  


# Loop through each column index and calculate Pearson correlation between each radiomic feature in Pyrads and survival endpoint
correlations_pearson_pyrads = []
p_values_pearson_pyrads = []
for i in range(1,df_pyrads_data.shape[1]):
    mask = ~np.isnan(df_pyrads_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_PercTotal'])
    corr, pvalue = pearsonr(df_pyrads_data.iloc[:, i][mask], df_IHC_data['CD8_PercTotal'][mask])
    correlations_pearson_pyrads.append(corr)
    p_values_pearson_pyrads.append(pvalue)

# Loop through each column index and calculate Pearson correlation between each radiomic feature in RaCat and survival endpoint
correlations_pearson_racat = []
p_values_pearson_racat = []
for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_PercTotal'])
    corr, pvalue = pearsonr(df_racat_data.iloc[:, i][mask], df_IHC_data['CD8_PercTotal'][mask])
    correlations_pearson_racat.append(corr)
    p_values_pearson_racat.append(pvalue)


# Loop through each column index and calculate Spearman correlation between each radiomic feature in RaCat and survival endpoint
correlations_spear_pyrads = []
p_values_spear_pyrads = []
for i in range(1,df_pyrads_data.shape[1]):
    mask = ~np.isnan(df_pyrads_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_PercTotal'])
    corr, pvalue = spearmanr(df_pyrads_data.iloc[:, i][mask], df_IHC_data['CD8_PercTotal'][mask])
    correlations_spear_pyrads.append(corr)
    p_values_spear_pyrads.append(pvalue)

correlations_spear_racat = []
p_values_spear_racat = []
for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_PercTotal'])
    corr, pvalue = spearmanr(df_racat_data.iloc[:, i][mask], df_IHC_data['CD8_PercTotal'][mask])
    correlations_spear_racat.append(corr)
    p_values_spear_racat.append(pvalue)


correlations_kendal_pyrads = []
p_values_kendal_pyrads = []
for i in range(1,df_pyrads_data.shape[1]):
    mask = ~np.isnan(df_pyrads_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_PercTotal'])
    corr, pvalue = kendalltau(df_pyrads_data.iloc[:, i][mask], df_IHC_data['CD8_PercTotal'][mask])
    correlations_kendal_pyrads.append(corr)
    p_values_kendal_pyrads.append(pvalue)

correlations_kendal_racat = []
p_values_kendal_racat = []
for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_PercTotal'])
    corr, pvalue = kendalltau(df_racat_data.iloc[:, i][mask], df_IHC_data['CD8_PercTotal'][mask])
    correlations_kendal_racat.append(corr)
    p_values_kendal_racat.append(pvalue)


"""
#function to calculate f-test
def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    nun = len(group1)-1
    dun = len(group2)-1
    p_value = 1-scipy.stats.f.cdf(f, nun, dun)
    return f, p_value


# Loop through each column index and calculate f-value and p-value from f-test
kendal_pyrads = []
p_values_ftest_pyrads = []
for i in range(1,df_pyrads_data.shape[1]):
    mask = ~np.isnan(df_pyrads_data.iloc[:, i]) & ~np.isnan(df_IHC_data['pfs_days'])
    fvalue, pvalue = f_test(df_pyrads_data.iloc[:, i][mask], df_IHC_data['pfs_days'][mask])
    ftest_pyrads.append(abs(fvalue))
    p_values_ftest_pyrads.append(pvalue)

ftest_racat = []
p_values_ftest_racat = []
for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['pfs_days'])
    fvalue, pvalue = f_test(df_racat_data.iloc[:, i][mask], df_IHC_data['pfs_days'][mask])
    ftest_racat.append(abs(fvalue))
    p_values_ftest_racat.append(pvalue)
"""

#dataframe of Pyrads correlation results
data_pyrads = {'corr_pyrads_spear':correlations_spear_pyrads[1:], 'pval_pyrads_spear':p_values_spear_pyrads[1:],
'corr_pyrads_pear':correlations_pearson_pyrads[1:], 'pval_pyrads_pear':p_values_pearson_pyrads[1:],
'corr_pyrads_kendal':correlations_kendal_pyrads[1:], 'pval_pyrads_kendal':p_values_kendal_pyrads[1:] }

#dataframe of Racat correlation results
data_racat = {'corr_racat_spear':correlations_spear_racat[1:], 'pval_racat_spear':p_values_spear_racat[1:],
'corr_racat_pear':correlations_pearson_racat[1:], 'pval_racat_pear':p_values_pearson_racat[1:],
'corr_racat_kendal':correlations_kendal_racat[1:], 'pval_racat_kendal':p_values_kendal_racat[1:] }

df_pyrads = pd.DataFrame(data_pyrads)
df_racat = pd.DataFrame(data_racat)

#label the significant correlations for each evaluator
df_pyrads['Spearman'] = df_pyrads['pval_pyrads_spear'] < 0.05
df_pyrads['Pearson'] = df_pyrads['pval_pyrads_pear'] < 0.05
df_pyrads['kendal'] = df_pyrads['pval_pyrads_kendal'] < 0.05

df_racat['Spearman'] = df_racat['pval_racat_spear'] < 0.05
df_racat['Pearson'] = df_racat['pval_racat_pear'] < 0.05
df_racat['kendal'] = df_racat['pval_racat_kendal'] < 0.05

#save the results in excel files
writer = pd.ExcelWriter(os.path.join(main_dir, 'Results/data_analysis/final-analysis_new/evaluators_common_features_pyrads_CD8_new.xlsx'), engine='xlsxwriter')
wb  = writer.book
df_pyrads.to_excel(writer)    ## write into excel
wb.close()

writer = pd.ExcelWriter(os.path.join(main_dir, 'Results/data_analysis/final-analysis_new/evaluators_common_features_racat_CD8_new.xlsx'), engine='xlsxwriter')
wb  = writer.book
df_racat.to_excel(writer)    ## write into excel
wb.close()

