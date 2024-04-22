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
main_dir = Path('path of the main folder')

#read the features from an Excel file
rad_data = pd.read_excel(os.path.join(main_dir, 'radiomicsfeatures_pyrads_binwidth25.xlsx')) 
racat_data = pd.read_csv(os.path.join(main_dir, 'RaCat_concat_bin25.csv')) 

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

group_payrads_dir = Path('data_analysis/Pyrads')
group_racat_dir = Path('data_analysis/Racat')
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
