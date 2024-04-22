import pandas as pd
from scipy.stats import pearsonr, spearmanr
#from sklearn.feature_selection import f_regression, mutual_info_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import os
from pathlib import Path
import pyreadr
import scipy
from scipy import stats

#warnings.filterwarnings('ignore')
#set_printoptions(precision=3)

#path of the main folder
main_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ_projects/EGFR_prediction/test_samples/Test-Venkata/iucpq_data/')

#read the features from an Excel file
rad_data = pd.read_excel(os.path.join(main_dir, 'Results/DGX_output/radiomicsfeatures_pyrads_binwidth25.xlsx')) 
racat_data = pd.read_csv(os.path.join(main_dir, 'Results/PC_output/RaCat_concat_bin25.csv')) 

# select the float columns from rad_data
rad_data = rad_data.select_dtypes(include=[float])
racat_data = racat_data.select_dtypes(include=[float])

#place inf with nan in rad_data
rad_data.replace([np.inf, -np.inf], np.nan, inplace=True)
racat_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#drop the columns which include missing values from the dataset
rad_data.dropna(inplace=True, axis = 1)
racat_data.dropna(inplace=True, axis = 1)

#remove all features that are constant 
rad_data = rad_data.loc[:, rad_data.var() != 0.0]
racat_data = racat_data.loc[:, racat_data.var() != 0.0]
#print("Shape of the features after removing constant features =", np.shape(rad_data))
radiomics_headers_rads = list(rad_data.columns)
radiomics_headers_racat = list(racat_data.columns)

group_payrads_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ_projects/EGFR_prediction/test_samples/Test-Venkata/iucpq_data/Results/data_analysis/Pyrads')
group_racat_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ_projects/EGFR_prediction/test_samples/Test-Venkata/iucpq_data/Results/data_analysis/Racat')
group_pyrads_file = pd.read_excel(os.path.join(group_payrads_dir, 'bw25/rank_rads_bw25_os_days.xlsx')) 
group_racat_file = pd.read_excel(os.path.join(group_racat_dir, 'correlation_rads_bw25_os_days.xlsx')) 
group_pyrads = group_pyrads_file ['group1']
group_racat = group_racat_file ['class']


group_pyrads_original = group_pyrads_file['Unnamed: 0']
group_racat_original = group_racat_file['Unnamed: 0']


selected_features_pyrads = []
selected_features_racat = []

for i in range(len(group_pyrads_original)):
    if group_pyrads[i] == 'exp_first_order':  # Select values from array1 where corresponding value in array2 is glcm
        selected_features_pyrads.append(group_pyrads_original[i])

for i in range(len(group_racat_original)):
    if group_racat[i] == 'Intensity':  # Select values from array1 where corresponding value in array2 is glcm
        selected_features_racat.append(group_racat_original[i])


rad_data_glcm = rad_data[selected_features_pyrads]
racat_data_glcm = racat_data[selected_features_racat]

df1 = racat_data_glcm

df2 = rad_data_glcm



#corr, pvalue = spearmanr(df1[df1.columns[0]], df2[df2.columns[0]])
#print(corr)
#exit()
# assume df1 and df2 have the same number of rows
correlations = []
pvalues = []
for col1 in df1.columns:
    for col2 in df2.columns:
    	corr, pvalue = pearsonr(df1[col1], df2[col2])
    	correlations.append(corr)
    	pvalues.append(pvalue)

# create a new dataframe to store the results
result_df = pd.DataFrame({'correlation': correlations, 'p-value': pvalues})

#print(len(result_df['correlation']))

# reshape the dataframe to have one row per column pair
result_df_corr = result_df['correlation'].values.reshape((len(df1.columns), len(df2.columns)))
result_df_corr = pd.DataFrame(result_df_corr, columns=['col2_' + str(i) for i in range(len(df2.columns))],
                         index=['col1_' + str(i) for i in range(len(df1.columns))])

# reshape the dataframe to have one row per column pair
result_df_pvalue = result_df['p-value'].values.reshape((len(df1.columns), len(df2.columns)))
result_df_pvalue = pd.DataFrame(result_df_pvalue, columns=['col2_' + str(i) for i in range(len(df2.columns))],
                         index=['col1_' + str(i) for i in range(len(df1.columns))])

index_p_value_sig = np.ndarray.tolist(np.where(result_df['p-value']<0.05)[0])
corr_p_value_sig = result_df['correlation'][index_p_value_sig]
print(len(corr_p_value_sig))



#corr_features_ind = corr_matrix_Textural.index

plt.figure(figsize=(10,7.5), dpi=200)
"""
#ax = sns.heatmap(result_df_corr ,annot=result_df_pvalue, annot_kws={"size":50}, cmap='YlGnBu', vmin=0,vmax=max(result_df['correlation']))
ax = sns.heatmap(result_df_corr,
            mask=result_df_pvalue < 0.05,
            linewidth=0.5,
            annot=result_df_pvalue, annot_kws={"size":140}, cmap='YlGnBu', vmin=0,vmax=max(result_df['correlation']))

ax = sns.heatmap(result_df_corr,
            mask=result_df_pvalue >= 0.05,
            annot_kws={"size":140, "style": "italic", "weight": "bold", "color": "red"},
            linewidth=0.5,
            annot=result_df_pvalue,
            cbar=False,
            cmap='YlGnBu')

"""

ax = sns.heatmap(result_df_corr,
            linewidth=1, linecolor="grey",
           cmap="YlGnBu", vmin=0,vmax=max(result_df['correlation']))

ax = sns.heatmap(result_df_corr,
            mask=result_df_pvalue >= 0.05,
            linewidth=0.5,
            annot=False,
            cbar=False,
            cmap='Reds')


# use matplotlib.colorbar.Colorbar object
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
#plt.xticks(fontsize=10, ticks = np.arange(1, len(df1.columns)), labels = np.arange(1, len(df1.columns)), rotation = 90)
plt.xticks(fontsize=6, ticks = np.arange(len(selected_features_pyrads))+0.5, labels = selected_features_pyrads, rotation = 45, ha="right")
#plt.yticks(fontsize=8, ticks = np.arange(1, len(df2.columns)), labels = np.arange(1, len(df2.columns)))
plt.yticks(fontsize=6, ticks = np.arange(len(selected_features_racat))+0.5, labels = selected_features_racat, rotation = 0)
plt.ylabel("Intensity-based radiomic features from Racat", fontsize= 10, labelpad = 5)
plt.xlabel("Exp first order radiomic features from Pyrads", fontsize= 10, labelpad = 5)
plt.title("The Pearson correlation between Intensity-based radiomic features from Pyrads and RaCat with bin width = 25", fontdict={'fontsize':13}, y= 1.02)
#plt.show()

plt.savefig(os.path.join(main_dir, 'Results/Plots/Plots-AAPM/heatmap_tools_exp_intensity_test.png'), bbox_inches='tight')
#plt.savefig(os.path.join(main_dir, 'Results/plots/heatmap_tools_glcm-glcm_davg.png'))