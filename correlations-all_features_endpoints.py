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
from scipy.stats import f_oneway, kruskal


#path of the main folder
main_dir = Path('path of the main folder')


#read the features from Excel files
#df_pyrads_data = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='radiomics_pyrads_bw25' )
df_pyrads_data = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_pyrads_bc128.xlsx'), sheet_name='radiomics_pyrads_bc128' )
df_clinical_data = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='clinical_data' )
df_IHC_data = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='IHC' )
#df_racat_data = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_racat.xlsx'), sheet_name='radiomics_racat')  
df_racat_data = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_pyrads_bc512.xlsx'), sheet_name='radiomics_pyrads_bc512')  
df_features_common = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='common-features')  
df_features_pyrads = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='features-Pyrads')  
df_features_racat = pd.read_excel(os.path.join(main_dir, '/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='features-RaCat')  

# Loop through each column index and calculate Pearson correlation between each radiomic feature in Pyrads and the endpoint
correlations_pearson_pyrads = []
p_values_pearson_pyrads = []
for i in range(1,df_pyrads_data.shape[1]):
    mask = ~np.isnan(df_pyrads_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_total'])
    corr, pvalue = pearsonr(df_pyrads_data.iloc[:, i][mask], df_IHC_data['CD8_total'][mask])
    correlations_pearson_pyrads.append(corr)
    p_values_pearson_pyrads.append(pvalue)

# Loop through each column index and calculate Pearson correlation between each radiomic feature in RaCat and the endpoint
correlations_pearson_racat = []
p_values_pearson_racat = []
for i in range(1,df_racat_data.shape[1]):
	mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_total'])
	corr, pvalue = pearsonr(df_racat_data.iloc[:, i][mask], df_IHC_data['CD8_total'][mask])
	correlations_pearson_racat.append(corr)
	p_values_pearson_racat.append(pvalue)


# Loop through each column index and calculate Spearman correlation between each radiomic feature in Pyrads and the endpoint
correlations_spear_pyrads = []
p_values_spear_pyrads = []
for i in range(1,df_pyrads_data.shape[1]):
    mask = ~np.isnan(df_pyrads_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_total'])
    corr, pvalue = spearmanr(df_pyrads_data.iloc[:, i][mask], df_IHC_data['CD8_total'][mask])
    correlations_spear_pyrads.append(corr)
    p_values_spear_pyrads.append(pvalue)
	
# Loop through each column index and calculate Spearman correlation between each radiomic feature in Pyrads and the endpoint
correlations_spear_racat = []
p_values_spear_racat = []
for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_total'])
    corr, pvalue = spearmanr(df_racat_data.iloc[:, i][mask], df_IHC_data['CD8_total'][mask])
    correlations_spear_racat.append(corr)
    p_values_spear_racat.append(pvalue)


#function to calculate f-test
"""
def f_test(group1, group2):
    f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
    nun = len(group1)-1
    dun = len(group2)-1
    p_value = 1-scipy.stats.f.cdf(f, nun, dun)
    return f, p_value
"""

# Loop through each column index and calculate kendall correlation between each radiomic feature in Pyrads and the endpoint
correlations_kendal_pyrads = []
p_values_kendal_pyrads = []
for i in range(1,df_pyrads_data.shape[1]):
    mask = ~np.isnan(df_pyrads_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_total'])
    corr, pvalue = kendalltau(df_pyrads_data.iloc[:, i][mask], df_IHC_data['CD8_total'][mask])
    correlations_kendal_pyrads.append(corr)
    p_values_kendal_pyrads.append(pvalue)

# Loop through each column index and calculate kendall correlation between each radiomic feature in racat and the endpoint
correlations_kendal_racat = []
p_values_kendal_racat = []
for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_total'])
    corr, pvalue = kendalltau(df_racat_data.iloc[:, i][mask], df_IHC_data['CD8_total'][mask])
    correlations_kendal_racat.append(corr)
    p_values_kendal_racat.append(pvalue)


#dataframe of Pyrads correlation results
data_pyrads = {'corr_pyrads_spear':correlations_spear_pyrads[0:], 'pval_pyrads_spear':p_values_spear_pyrads[0:],
'corr_pyrads_pear':correlations_pearson_pyrads[0:], 'pval_pyrads_pear':p_values_pearson_pyrads[0:],
'corr_pyrads_kendal':correlations_kendal_pyrads[0:], 'pval_pyrads_kendal':p_values_kendal_pyrads[0:] }

#dataframe of Racat correlation results
data_racat = {'corr_racat_spear':correlations_spear_racat[0:], 'pval_racat_spear':p_values_spear_racat[0:],
'corr_racat_pear':correlations_pearson_racat[0:], 'pval_racat_pear':p_values_pearson_racat[0:],
'corr_racat_kendal':correlations_kendal_racat[0:], 'pval_racat_kendal':p_values_kendal_racat[0:] }

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
writer = pd.ExcelWriter(os.path.join(main_dir, '/evaluators_all_features_pyrads_bc128_cd8.xlsx'), engine='xlsxwriter')
wb  = writer.book
df_pyrads.to_excel(writer)    ## write into excel
wb.close()

writer = pd.ExcelWriter(os.path.join(main_dir, '/evaluators_all_features_racat_cd8.xlsx'), engine='xlsxwriter')
wb  = writer.book
df_racat.to_excel(writer)    ## write into excel
wb.close()

