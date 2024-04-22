import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import os
import statistics

#path of the main folder
main_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ/EGFR_prediction/test_samples/iucpq-Venkata/')

#read the features from Excel files
df_pyrads_pvals_os = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_pyrads_os.xlsx'))
df_racat_pvals_os = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_racat_os.xlsx'))

df_pyrads_pvals_pfs = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_pyrads_pfs.xlsx'))
df_racat_pvals_pfs = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_racat_pfs.xlsx'))

df_pyrads_pvals_cd8 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_pyrads_cd8.xlsx'))
df_racat_pvals_cd8 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_racat_cd8.xlsx'))

df_features = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='common-features')  


ind_os_pyrads_spear = [i for i, x in enumerate(df_pyrads_pvals_os['Spearman']) if x]
features_os_pyrads_spear = df_features['features'][ind_os_pyrads_spear]
#print(features_os_pyrads_spear)
print(len(ind_os_pyrads_spear))


ind_os_racat_spear = [i for i, x in enumerate(df_racat_pvals_os['Spearman']) if x]
features_os_racat_spear = df_features['features'][ind_os_racat_spear]
#print(features_os_racat_spear)
print(len(ind_os_racat_spear))
print(features_os_pyrads_spear[features_os_pyrads_spear.isin(features_os_racat_spear)])

ind_os_pyrads_pear = [i for i, x in enumerate(df_pyrads_pvals_os['Pearson']) if x]
features_os_pyrads_pear = df_features['features'][ind_os_pyrads_pear]
print(len(ind_os_pyrads_pear))


ind_os_racat_pear = [i for i, x in enumerate(df_racat_pvals_os['Pearson']) if x]
features_os_racat_pear = df_features['features'][ind_os_racat_pear]
print(len(ind_os_racat_pear))
print(features_os_pyrads_pear[features_os_pyrads_pear.isin(features_os_racat_pear)])

ind_pfs_pyrads_spear = [i for i, x in enumerate(df_pyrads_pvals_pfs['Spearman']) if x]
features_pfs_pyrads_spear = df_features['features'][ind_pfs_pyrads_spear]
print(len(ind_pfs_pyrads_spear))

ind_pfs_racat_spear = [i for i, x in enumerate(df_racat_pvals_pfs['Spearman']) if x]
features_pfs_racat_spear = df_features['features'][ind_pfs_racat_spear]
print(len(ind_pfs_racat_spear))
print(features_pfs_pyrads_spear[features_pfs_pyrads_spear.isin(features_pfs_racat_spear)])


ind_pfs_pyrads_pear = [i for i, x in enumerate(df_pyrads_pvals_pfs['Pearson']) if x]
features_pfs_pyrads_pear = df_features['features'][ind_pfs_pyrads_pear]
print(len(ind_pfs_pyrads_pear))


ind_pfs_racat_pear = [i for i, x in enumerate(df_racat_pvals_pfs['Pearson']) if x]
features_pfs_racat_pear = df_features['features'][ind_pfs_racat_pear]
print(len(ind_pfs_racat_pear))
print(features_pfs_pyrads_pear[features_pfs_pyrads_pear.isin(features_pfs_racat_pear)])

ind_cd8_pyrads_spear = [i for i, x in enumerate(df_pyrads_pvals_cd8['Spearman']) if x]
features_cd8_pyrads_spear = df_features['features'][ind_cd8_pyrads_spear]
print(len(ind_cd8_pyrads_spear))

ind_cd8_racat_spear = [i for i, x in enumerate(df_racat_pvals_cd8['Spearman']) if x]
features_cd8_racat_spear = df_features['features'][ind_cd8_racat_spear]
print(len(ind_cd8_racat_spear))

print(features_cd8_pyrads_spear[features_cd8_pyrads_spear.isin(features_cd8_racat_spear)])

ind_cd8_pyrads_pear = [i for i, x in enumerate(df_pyrads_pvals_cd8['Pearson']) if x]
features_cd8_pyrads_pear = df_features['features'][ind_cd8_pyrads_pear]
print(len(ind_cd8_pyrads_pear))

ind_cd8_racat_pear = [i for i, x in enumerate(df_racat_pvals_cd8['Pearson']) if x]
features_cd8_racat_pear = df_features['features'][ind_cd8_racat_pear]
print(len(ind_cd8_racat_pear))
print(features_cd8_pyrads_pear[features_cd8_pyrads_spear.isin(features_cd8_racat_pear)])