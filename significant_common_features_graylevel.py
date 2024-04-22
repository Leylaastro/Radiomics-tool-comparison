import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import os
import statistics
import seaborn as sns

#path of the main folder
main_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ/EGFR_prediction/test_samples/iucpq-Venkata/')

#read the features from Excel files
df_pyradsbw1_pvals_pfs = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bw25_pfs.xlsx'))
df_pyradsbw2_pvals_pfs = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bw50_pfs.xlsx'))

df_pyradsbw1_pvals_cd8 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bw25_cd8.xlsx'))
df_pyradsbw2_pvals_cd8 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bw50_cd8.xlsx'))

df_pyradsbw1_pvals_pdl1 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bw25_pdl1.xlsx'))
df_pyradsbw2_pvals_pdl1 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bw50_pdl1.xlsx'))


df_pyradsbc1_pvals_pfs = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bc128_pfs.xlsx'))
df_pyradsbc2_pvals_pfs = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bc512_pfs.xlsx'))

df_pyradsbc1_pvals_cd8 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bc128_cd8.xlsx'))
df_pyradsbc2_pvals_cd8 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bc512_cd8.xlsx'))

df_pyradsbc1_pvals_pdl1 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bc128_pdl1.xlsx'))
df_pyradsbc2_pvals_pdl1 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_all_features_pyrads_bc512_pdl1.xlsx'))

df_features = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='features-Pyrads')  


ind_pfs_pyradsbw1_pear = [i for i, x in enumerate(df_pyradsbw1_pvals_pfs['Pearson']) if x]
features_pfs_pyradsbw1_pear = df_features['features'][ind_pfs_pyradsbw1_pear]
print(len(ind_pfs_pyradsbw1_pear))

ind_pfs_pyradsbw2_pear = [i for i, x in enumerate(df_pyradsbw2_pvals_pfs['Pearson']) if x]
features_pfs_pyradsbw2_pear = df_features['features'][ind_pfs_pyradsbw2_pear]
print(len(ind_pfs_pyradsbw2_pear))

ind_pfs_pyradsbw_pear = list(set(ind_pfs_pyradsbw1_pear).intersection(ind_pfs_pyradsbw2_pear))
print(len(ind_pfs_pyradsbw_pear))

ind_pfs_pyradsbc1_pear = [i for i, x in enumerate(df_pyradsbc1_pvals_pfs['Pearson']) if x]
features_pfs_pyradsbc1_pear = df_features['features'][ind_pfs_pyradsbc1_pear]
print(len(ind_pfs_pyradsbc1_pear))

ind_pfs_pyradsbc2_pear = [i for i, x in enumerate(df_pyradsbc2_pvals_pfs['Pearson']) if x]
features_pfs_pyradsbc2_pear = df_features['features'][ind_pfs_pyradsbc2_pear]
print(len(ind_pfs_pyradsbc2_pear))
ind_pfs_pyradsbc_pear = list(set(ind_pfs_pyradsbc1_pear).intersection(ind_pfs_pyradsbc2_pear))
print(len(ind_pfs_pyradsbc_pear))

ind_pfs_pyrads_pear =  list(set(ind_pfs_pyradsbw_pear).intersection(ind_pfs_pyradsbc_pear))
print(len(ind_pfs_pyrads_pear))


ind_pfs_pyradsbw1_spear = [i for i, x in enumerate(df_pyradsbw1_pvals_pfs['Spearman']) if x]
features_pfs_pyradsbw1_spear = df_features['features'][ind_pfs_pyradsbw1_spear]
print(len(ind_pfs_pyradsbw1_spear))

ind_pfs_pyradsbw2_spear = [i for i, x in enumerate(df_pyradsbw2_pvals_pfs['Spearman']) if x]
features_pfs_pyradsbw2_spear = df_features['features'][ind_pfs_pyradsbw2_spear]
print(len(ind_pfs_pyradsbw2_spear))

ind_pfs_pyradsbw_spear = list(set(ind_pfs_pyradsbw1_spear).intersection(ind_pfs_pyradsbw2_spear))
print(len(ind_pfs_pyradsbw_spear))

ind_pfs_pyradsbc1_spear = [i for i, x in enumerate(df_pyradsbc1_pvals_pfs['Spearman']) if x]
features_pfs_pyradsbc1_spear = df_features['features'][ind_pfs_pyradsbc1_spear]
print(len(ind_pfs_pyradsbc1_spear))

ind_pfs_pyradsbc2_spear = [i for i, x in enumerate(df_pyradsbc2_pvals_pfs['Spearman']) if x]
features_pfs_pyradsbc2_spear = df_features['features'][ind_pfs_pyradsbc2_spear]
print(len(ind_pfs_pyradsbc2_spear))
ind_pfs_pyradsbc_spear = list(set(ind_pfs_pyradsbc1_spear).intersection(ind_pfs_pyradsbc2_spear))
print(len(ind_pfs_pyradsbc_spear))

ind_pfs_pyrads_spear =  list(set(ind_pfs_pyradsbw_spear).intersection(ind_pfs_pyradsbc_spear))
print(len(ind_pfs_pyrads_spear))



ind_pfs_pyradsbw1_kendal = [i for i, x in enumerate(df_pyradsbw1_pvals_pfs['kendal']) if x]
features_pfs_pyradsbw1_kendal = df_features['features'][ind_pfs_pyradsbw1_kendal]
print(len(ind_pfs_pyradsbw1_kendal))

ind_pfs_pyradsbw2_kendal = [i for i, x in enumerate(df_pyradsbw2_pvals_pfs['kendal']) if x]
features_pfs_pyradsbw2_kendal = df_features['features'][ind_pfs_pyradsbw2_kendal]
print(len(ind_pfs_pyradsbw2_kendal))

ind_pfs_pyradsbw_kendal = list(set(ind_pfs_pyradsbw1_kendal).intersection(ind_pfs_pyradsbw2_kendal))
print(len(ind_pfs_pyradsbw_kendal))

ind_pfs_pyradsbc1_kendal = [i for i, x in enumerate(df_pyradsbc1_pvals_pfs['kendal']) if x]
features_pfs_pyradsbc1_kendal = df_features['features'][ind_pfs_pyradsbc1_kendal]
print(len(ind_pfs_pyradsbc1_kendal))

ind_pfs_pyradsbc2_kendal = [i for i, x in enumerate(df_pyradsbc2_pvals_pfs['kendal']) if x]
features_pfs_pyradsbc2_kendal = df_features['features'][ind_pfs_pyradsbc2_kendal]
print(len(ind_pfs_pyradsbc2_kendal))
ind_pfs_pyradsbc_kendal = list(set(ind_pfs_pyradsbc1_kendal).intersection(ind_pfs_pyradsbc2_kendal))
print(len(ind_pfs_pyradsbc_kendal))

ind_pfs_pyrads_kendal =  list(set(ind_pfs_pyradsbw_kendal).intersection(ind_pfs_pyradsbc_kendal))
print(len(ind_pfs_pyrads_kendal))


ind_pfs_pyrads = list(set(ind_pfs_pyrads_pear).intersection(ind_pfs_pyrads_spear).intersection(ind_pfs_pyrads_kendal))
print(len(ind_pfs_pyrads))


ind_pdl1_pyradsbw1_pear = [i for i, x in enumerate(df_pyradsbw1_pvals_pdl1['Pearson']) if x]
features_pdl1_pyradsbw1_pear = df_features['features'][ind_pdl1_pyradsbw1_pear]
print(len(ind_pdl1_pyradsbw1_pear))

ind_pdl1_pyradsbw2_pear = [i for i, x in enumerate(df_pyradsbw2_pvals_pdl1['Pearson']) if x]
features_pdl1_pyradsbw2_pear = df_features['features'][ind_pdl1_pyradsbw2_pear]
print(len(ind_pdl1_pyradsbw2_pear))

ind_pdl1_pyradsbw_pear = list(set(ind_pdl1_pyradsbw1_pear).intersection(ind_pdl1_pyradsbw2_pear))
print(len(ind_pdl1_pyradsbw_pear))

ind_pdl1_pyradsbc1_pear = [i for i, x in enumerate(df_pyradsbc1_pvals_pdl1['Pearson']) if x]
features_pdl1_pyradsbc1_pear = df_features['features'][ind_pdl1_pyradsbc1_pear]
print(len(ind_pdl1_pyradsbc1_pear))

ind_pdl1_pyradsbc2_pear = [i for i, x in enumerate(df_pyradsbc2_pvals_pdl1['Pearson']) if x]
features_pdl1_pyradsbc2_pear = df_features['features'][ind_pdl1_pyradsbc2_pear]
print(len(ind_pdl1_pyradsbc2_pear))
ind_pdl1_pyradsbc_pear = list(set(ind_pdl1_pyradsbc1_pear).intersection(ind_pdl1_pyradsbc2_pear))
print(len(ind_pdl1_pyradsbc_pear))

ind_pdl1_pyrads_pear =  list(set(ind_pdl1_pyradsbw_pear).intersection(ind_pdl1_pyradsbc_pear))
print(len(ind_pdl1_pyrads_pear))


ind_pdl1_pyradsbw1_spear = [i for i, x in enumerate(df_pyradsbw1_pvals_pdl1['Spearman']) if x]
features_pdl1_pyradsbw1_spear = df_features['features'][ind_pdl1_pyradsbw1_spear]
print(len(ind_pdl1_pyradsbw1_spear))

ind_pdl1_pyradsbw2_spear = [i for i, x in enumerate(df_pyradsbw2_pvals_pdl1['Spearman']) if x]
features_pdl1_pyradsbw2_spear = df_features['features'][ind_pdl1_pyradsbw2_spear]
print(len(ind_pdl1_pyradsbw2_spear))

ind_pdl1_pyradsbw_spear = list(set(ind_pdl1_pyradsbw1_spear).intersection(ind_pdl1_pyradsbw2_spear))
print(len(ind_pdl1_pyradsbw_spear))

ind_pdl1_pyradsbc1_spear = [i for i, x in enumerate(df_pyradsbc1_pvals_pdl1['Spearman']) if x]
features_pdl1_pyradsbc1_spear = df_features['features'][ind_pdl1_pyradsbc1_spear]
print(len(ind_pdl1_pyradsbc1_spear))

ind_pdl1_pyradsbc2_spear = [i for i, x in enumerate(df_pyradsbc2_pvals_pdl1['Spearman']) if x]
features_pdl1_pyradsbc2_spear = df_features['features'][ind_pdl1_pyradsbc2_spear]
print(len(ind_pdl1_pyradsbc2_spear))
ind_pdl1_pyradsbc_spear = list(set(ind_pdl1_pyradsbc1_spear).intersection(ind_pdl1_pyradsbc2_spear))
print(len(ind_pdl1_pyradsbc_spear))

ind_pdl1_pyrads_spear =  list(set(ind_pdl1_pyradsbw_spear).intersection(ind_pdl1_pyradsbc_spear))
print(len(ind_pdl1_pyrads_spear))



ind_pdl1_pyradsbw1_kendal = [i for i, x in enumerate(df_pyradsbw1_pvals_pdl1['kendal']) if x]
features_pdl1_pyradsbw1_kendal = df_features['features'][ind_pdl1_pyradsbw1_kendal]
print(len(ind_pdl1_pyradsbw1_kendal))

ind_pdl1_pyradsbw2_kendal = [i for i, x in enumerate(df_pyradsbw2_pvals_pdl1['kendal']) if x]
features_pdl1_pyradsbw2_kendal = df_features['features'][ind_pdl1_pyradsbw2_kendal]
print(len(ind_pdl1_pyradsbw2_kendal))

ind_pdl1_pyradsbw_kendal = list(set(ind_pdl1_pyradsbw1_kendal).intersection(ind_pdl1_pyradsbw2_kendal))
print(len(ind_pdl1_pyradsbw_kendal))

ind_pdl1_pyradsbc1_kendal = [i for i, x in enumerate(df_pyradsbc1_pvals_pdl1['kendal']) if x]
features_pdl1_pyradsbc1_kendal = df_features['features'][ind_pdl1_pyradsbc1_kendal]
print(len(ind_pdl1_pyradsbc1_kendal))

ind_pdl1_pyradsbc2_kendal = [i for i, x in enumerate(df_pyradsbc2_pvals_pdl1['kendal']) if x]
features_pdl1_pyradsbc2_kendal = df_features['features'][ind_pdl1_pyradsbc2_kendal]
print(len(ind_pdl1_pyradsbc2_kendal))
ind_pdl1_pyradsbc_kendal = list(set(ind_pdl1_pyradsbc1_kendal).intersection(ind_pdl1_pyradsbc2_kendal))
print(len(ind_pdl1_pyradsbc_kendal))

ind_pdl1_pyrads_kendal =  list(set(ind_pdl1_pyradsbw_kendal).intersection(ind_pdl1_pyradsbc_kendal))
print(len(ind_pdl1_pyrads_kendal))


ind_pdl1_pyrads = list(set(ind_pdl1_pyrads_pear).intersection(ind_pdl1_pyrads_spear).intersection(ind_pdl1_pyrads_kendal))
print(len(ind_pdl1_pyrads))


ind_cd8_pyradsbw1_pear = [i for i, x in enumerate(df_pyradsbw1_pvals_cd8['Pearson']) if x]
features_cd8_pyradsbw1_pear = df_features['features'][ind_cd8_pyradsbw1_pear]
print(len(ind_cd8_pyradsbw1_pear))

ind_cd8_pyradsbw2_pear = [i for i, x in enumerate(df_pyradsbw2_pvals_cd8['Pearson']) if x]
features_cd8_pyradsbw2_pear = df_features['features'][ind_cd8_pyradsbw2_pear]
print(len(ind_cd8_pyradsbw2_pear))

ind_cd8_pyradsbw_pear = list(set(ind_cd8_pyradsbw1_pear).intersection(ind_cd8_pyradsbw2_pear))
print(len(ind_cd8_pyradsbw_pear))

ind_cd8_pyradsbc1_pear = [i for i, x in enumerate(df_pyradsbc1_pvals_cd8['Pearson']) if x]
features_cd8_pyradsbc1_pear = df_features['features'][ind_cd8_pyradsbc1_pear]
print(len(ind_cd8_pyradsbc1_pear))

ind_cd8_pyradsbc2_pear = [i for i, x in enumerate(df_pyradsbc2_pvals_cd8['Pearson']) if x]
features_cd8_pyradsbc2_pear = df_features['features'][ind_cd8_pyradsbc2_pear]
print(len(ind_cd8_pyradsbc2_pear))
ind_cd8_pyradsbc_pear = list(set(ind_cd8_pyradsbc1_pear).intersection(ind_cd8_pyradsbc2_pear))
print(len(ind_cd8_pyradsbc_pear))

ind_cd8_pyrads_pear =  list(set(ind_cd8_pyradsbw_pear).intersection(ind_cd8_pyradsbc_pear))
print(len(ind_cd8_pyrads_pear))


ind_cd8_pyradsbw1_spear = [i for i, x in enumerate(df_pyradsbw1_pvals_cd8['Spearman']) if x]
features_cd8_pyradsbw1_spear = df_features['features'][ind_cd8_pyradsbw1_spear]
print(len(ind_cd8_pyradsbw1_spear))

ind_cd8_pyradsbw2_spear = [i for i, x in enumerate(df_pyradsbw2_pvals_cd8['Spearman']) if x]
features_cd8_pyradsbw2_spear = df_features['features'][ind_cd8_pyradsbw2_spear]
print(len(ind_cd8_pyradsbw2_spear))

ind_cd8_pyradsbw_spear = list(set(ind_cd8_pyradsbw1_spear).intersection(ind_cd8_pyradsbw2_spear))
print(len(ind_cd8_pyradsbw_spear))

ind_cd8_pyradsbc1_spear = [i for i, x in enumerate(df_pyradsbc1_pvals_cd8['Spearman']) if x]
features_cd8_pyradsbc1_spear = df_features['features'][ind_cd8_pyradsbc1_spear]
print(len(ind_cd8_pyradsbc1_spear))

ind_cd8_pyradsbc2_spear = [i for i, x in enumerate(df_pyradsbc2_pvals_cd8['Spearman']) if x]
features_cd8_pyradsbc2_spear = df_features['features'][ind_cd8_pyradsbc2_spear]
print(len(ind_cd8_pyradsbc2_spear))
ind_cd8_pyradsbc_spear = list(set(ind_cd8_pyradsbc1_spear).intersection(ind_cd8_pyradsbc2_spear))
print(len(ind_cd8_pyradsbc_spear))

ind_cd8_pyrads_spear =  list(set(ind_cd8_pyradsbw_spear).intersection(ind_cd8_pyradsbc_spear))
print(len(ind_cd8_pyrads_spear))



ind_cd8_pyradsbw1_kendal = [i for i, x in enumerate(df_pyradsbw1_pvals_cd8['kendal']) if x]
features_cd8_pyradsbw1_kendal = df_features['features'][ind_cd8_pyradsbw1_kendal]
print(len(ind_cd8_pyradsbw1_kendal))

ind_cd8_pyradsbw2_kendal = [i for i, x in enumerate(df_pyradsbw2_pvals_cd8['kendal']) if x]
features_cd8_pyradsbw2_kendal = df_features['features'][ind_cd8_pyradsbw2_kendal]
print(len(ind_cd8_pyradsbw2_kendal))

ind_cd8_pyradsbw_kendal = list(set(ind_cd8_pyradsbw1_kendal).intersection(ind_cd8_pyradsbw2_kendal))
print(len(ind_cd8_pyradsbw_kendal))

ind_cd8_pyradsbc1_kendal = [i for i, x in enumerate(df_pyradsbc1_pvals_cd8['kendal']) if x]
features_cd8_pyradsbc1_kendal = df_features['features'][ind_cd8_pyradsbc1_kendal]
print(len(ind_cd8_pyradsbc1_kendal))

ind_cd8_pyradsbc2_kendal = [i for i, x in enumerate(df_pyradsbc2_pvals_cd8['kendal']) if x]
features_cd8_pyradsbc2_kendal = df_features['features'][ind_cd8_pyradsbc2_kendal]
print(len(ind_cd8_pyradsbc2_kendal))
ind_cd8_pyradsbc_kendal = list(set(ind_cd8_pyradsbc1_kendal).intersection(ind_cd8_pyradsbc2_kendal))
print(len(ind_cd8_pyradsbc_kendal))

ind_cd8_pyrads_kendal =  list(set(ind_cd8_pyradsbw_kendal).intersection(ind_cd8_pyradsbc_kendal))
print(len(ind_cd8_pyrads_kendal))


ind_cd8_pyrads = list(set(ind_cd8_pyrads_pear).intersection(ind_cd8_pyrads_spear).intersection(ind_cd8_pyrads_kendal))
print(len(ind_cd8_pyrads))


#barplot 
features_pfs_pyrads = df_features['features'][ind_pfs_pyrads]
group_features_pfs_pyrads = df_features['group'][ind_pfs_pyrads]

subgroup_features_pfs_pyrads = df_features['group2'][ind_pfs_pyrads]
plt.figure()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
sns.countplot(x=group_features_pfs_pyrads, color= "cornflowerblue", order=group_features_pfs_pyrads.value_counts().index, saturation=0.75)
plt.ylabel('Count', fontsize =25, labelpad = 15)
plt.xlabel('Group of features', fontsize =25, labelpad = 15)
plt.show()

plt.figure()
plt.xticks(fontsize=20, ticks = np.arange(len(subgroup_features_pfs_pyrads))+0.5, labels = subgroup_features_pfs_pyrads, rotation = 45)
plt.yticks(fontsize=25)
sns.countplot(x=subgroup_features_pfs_pyrads, color = 'brown', order=subgroup_features_pfs_pyrads.value_counts().index, saturation=0.75)
plt.ylabel('Count', fontsize =25, labelpad = 15)
plt.xlabel('Sub-group of features', fontsize =25, labelpad = 15)


plt.show()

