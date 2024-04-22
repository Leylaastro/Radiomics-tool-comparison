import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path
import os
import statistics


#path of the main folder
main_dir = Path('path of main folder')

#read the features from an Excel file
df_pyrads_data = pd.read_excel(os.path.join(main_dir, 'clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='pyrads-common-racat' )
df_racat_data = pd.read_excel(os.path.join(main_dir, 'clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='racat-common-pyrads')  
df_features = pd.read_excel(os.path.join(main_dir, 'clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='common-features')  

#label of common features
features = df_features['group']


# Loop through each column index and calculate Pearson correlation
correlations = []
p_values = []
for i in range(1,df_pyrads_data.shape[1]):
    corr, pvalue = pearsonr(df_pyrads_data.iloc[:, i], df_racat_data.iloc[:, i])
    correlations.append(abs(corr))
    p_values.append(pvalue)

#categorze the correlations based on the groups of the features
label_dict = {}

for label, num in zip(features, correlations ):
    if label not in label_dict:
        label_dict[label] = [num]
    else:
        label_dict[label].append(num)


mean_corr = []
std_corr = []

#calculate the mean and SD for each item in label_dict 
for label, nums in label_dict.items():
    mean = statistics.mean(nums)
    if len(nums)>1:
    	std_dev = statistics.stdev(nums)
    else:
    	std_dev=0
    mean_corr.append(mean)
    std_corr.append(std_dev)

# Create lists for the plot
fetaure_group = ['Shape', 'Intensity', 'Texture']
x_pos = np.arange(len(fetaure_group))
# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, mean_corr[1:], yerr=std_corr[1:], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Correlation (Pearson)', fontsize =20)
ax.set_xticks(x_pos)
ax.set_xticklabels(fetaure_group, fontsize = 15)
ax.yaxis.grid(True)

# show the figure
plt.tight_layout()
plt.show()
