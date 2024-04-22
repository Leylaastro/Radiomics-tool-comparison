import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import os
import statistics


#path of the main folder
main_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ/EGFR_prediction/test_samples/iucpq-Venkata/')
#read the features from Excel files
#df_pyrads_data_1 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='pyrads-common-racat' )
df_pyrads_data_1 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='radiomics_pyrads_bw25' )
df_clinical_data = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='clinical_data' )
#df_racat_data = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='racat-common-pyrads')  
df_racat_data = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw50.xlsx'), sheet_name='radiomics_pyrads_bw50')  
#df_features = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='common-features')  
df_features = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='features-Pyrads')  
df_IHC_data = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw25.xlsx'), sheet_name='IHC' )
#common features between Pyrads and Racat
features = df_features['group']

#another way to find and read common features between two platforms
#find the name of common features between RaCat and Pyrads from Pyrads
#features_pyrads = df_pyrads_data_1.columns
#df_pyrads_data_2_original = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/clinical_IHC_individual_inter_with_rads_pyrads_bw50.xlsx'), sheet_name='radiomics_pyrads_bw50' )
#df_pyrads_data_2 = df_pyrads_data_2_original [features_pyrads]

# Loop through each column index and calculate Pearson correlation
correlations_pyrads_pfs_spear = []
p_values_pyrads_pfs_spear = []

correlations_racat_pfs_spear = []
p_values_racat_pfs_spear = []

correlations_pyrads_os_spear = []
p_values_pyrads_os_spear = []

correlations_racat_os_spear = []
p_values_racat_os_spear = []

correlations_pyrads_cd8_spear = []
p_values_pyrads_cd8_spear = []

correlations_racat_cd8_spear = []
p_values_racat_cd8_spear = []


correlations_pyrads_pfs_pear = []
p_values_pyrads_pfs_pear = []

correlations_racat_pfs_pear = []
p_values_racat_pfs_pear = []

correlations_pyrads_os_pear = []
p_values_pyrads_os_pear = []

correlations_racat_os_pear = []
p_values_racat_os_pear = []

correlations_pyrads_cd8_pear = []
p_values_pyrads_cd8_pear = []

correlations_racat_cd8_pear = []
p_values_racat_cd8_pear = []


for i in range(1,df_pyrads_data_1.shape[1]):
	mask = ~np.isnan(df_pyrads_data_1.iloc[:, i]) & ~np.isnan(df_clinical_data['pfs_days'])
	corr, pvalue = spearmanr(df_pyrads_data_1.iloc[:, i][mask], df_clinical_data['pfs_days'][mask])
	correlations_pyrads_pfs_spear.append(abs(corr))
	p_values_pyrads_pfs_spear.append(pvalue)

for i in range(1,df_racat_data.shape[1]):
	mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_clinical_data['pfs_days'])
	corr, pvalue = spearmanr(df_racat_data.iloc[:, i][mask], df_clinical_data['pfs_days'][mask])
	correlations_racat_pfs_spear.append(abs(corr))
	p_values_racat_pfs_spear.append(pvalue)

for i in range(1,df_pyrads_data_1.shape[1]):
    mask = ~np.isnan(df_pyrads_data_1.iloc[:, i]) & ~np.isnan(df_clinical_data['os_days'])
    corr, pvalue = spearmanr(df_pyrads_data_1.iloc[:, i][mask], df_clinical_data['os_days'][mask])
    correlations_pyrads_os_spear.append(abs(corr))
    p_values_pyrads_os_spear.append(pvalue)

for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_clinical_data['os_days'])
    corr, pvalue = spearmanr(df_racat_data.iloc[:, i][mask], df_clinical_data['os_days'][mask])
    correlations_racat_os_spear.append(abs(corr))
    p_values_racat_os_spear.append(pvalue)


for i in range(1,df_pyrads_data_1.shape[1]):
    mask = ~np.isnan(df_pyrads_data_1.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_tumeur'])
    corr, pvalue = spearmanr(df_pyrads_data_1.iloc[:, i][mask], df_IHC_data['CD8_tumeur'][mask])
    correlations_pyrads_cd8_spear.append(abs(corr))
    p_values_pyrads_cd8_spear.append(pvalue)

for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_tumeur'])
    corr, pvalue = spearmanr(df_racat_data.iloc[:, i][mask], df_IHC_data['CD8_tumeur'][mask])
    correlations_racat_cd8_spear.append(abs(corr))
    p_values_racat_cd8_spear.append(pvalue)



for i in range(1,df_pyrads_data_1.shape[1]):
    mask = ~np.isnan(df_pyrads_data_1.iloc[:, i]) & ~np.isnan(df_clinical_data['pfs_days'])
    corr, pvalue = pearsonr(df_pyrads_data_1.iloc[:, i][mask], df_clinical_data['pfs_days'][mask])
    correlations_pyrads_pfs_pear.append(abs(corr))
    p_values_pyrads_pfs_pear.append(pvalue)

for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_clinical_data['pfs_days'])
    corr, pvalue = pearsonr(df_racat_data.iloc[:, i][mask], df_clinical_data['pfs_days'][mask])
    correlations_racat_pfs_pear.append(abs(corr))
    p_values_racat_pfs_pear.append(pvalue)

for i in range(1,df_pyrads_data_1.shape[1]):
    mask = ~np.isnan(df_pyrads_data_1.iloc[:, i]) & ~np.isnan(df_clinical_data['os_days'])
    corr, pvalue = pearsonr(df_pyrads_data_1.iloc[:, i][mask], df_clinical_data['os_days'][mask])
    correlations_pyrads_os_pear.append(abs(corr))
    p_values_pyrads_os_pear.append(pvalue)

for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_clinical_data['os_days'])
    corr, pvalue = pearsonr(df_racat_data.iloc[:, i][mask], df_clinical_data['os_days'][mask])
    correlations_racat_os_pear.append(abs(corr))
    p_values_racat_os_pear.append(pvalue)


for i in range(1,df_pyrads_data_1.shape[1]):
    mask = ~np.isnan(df_pyrads_data_1.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_tumeur'])
    corr, pvalue = pearsonr(df_pyrads_data_1.iloc[:, i][mask], df_IHC_data['CD8_tumeur'][mask])
    correlations_pyrads_cd8_pear.append(abs(corr))
    p_values_pyrads_cd8_pear.append(pvalue)

for i in range(1,df_racat_data.shape[1]):
    mask = ~np.isnan(df_racat_data.iloc[:, i]) & ~np.isnan(df_IHC_data['CD8_tumeur'])
    corr, pvalue = pearsonr(df_racat_data.iloc[:, i][mask], df_IHC_data['CD8_tumeur'][mask])
    correlations_racat_cd8_pear.append(abs(corr))
    p_values_racat_cd8_pear.append(pvalue)




pvals1_pyrads_os_spear = np.array(p_values_pyrads_os_spear[1:])
pvals2_racat_os_spear = np.array(p_values_racat_os_spear[1:])

pvals1_pyrads_os_pear = np.array(p_values_pyrads_os_pear[1:])
pvals2_racat_os_pear = np.array(p_values_racat_os_pear[1:])

pvals1_pyrads_pfs_spear = np.array(p_values_pyrads_pfs_spear[1:])
pvals2_racat_pfs_spear = np.array(p_values_racat_pfs_spear[1:])

pvals1_pyrads_pfs_pear = np.array(p_values_pyrads_pfs_pear[1:])
pvals2_racat_pfs_pear = np.array(p_values_racat_pfs_pear[1:])

pvals1_pyrads_cd8_spear = np.array(p_values_pyrads_cd8_spear[1:])
pvals2_racat_cd8_spear = np.array(p_values_racat_cd8_spear[1:])

pvals1_pyrads_cd8_pear = np.array(p_values_pyrads_cd8_pear[1:])
pvals2_racat_cd8_pear = np.array(p_values_racat_cd8_pear[1:])


"""
def fdr_bh(pvals, alpha=0.05):
 
    pvals = np.asarray(pvals)
    n = len(pvals)
    idx = np.argsort(pvals)
    pvals_sorted = pvals[idx]
    qvals = np.zeros(n)
    for i in range(n):
        if i == 0:
            qvals[i] = 0
        else:
            qvals[i] = np.min((n/i) * alpha * pvals_sorted[:i+1])
        if qvals[i] > 1:
            qvals[i] = 1
    return qvals[idx]

"""
def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]




pvals1_corrected_pyrads_os_spear = p_adjust_bh(pvals1_pyrads_os_spear)
for i, val in enumerate(pvals1_corrected_pyrads_os_spear):
	if val == 0.:
		pvals1_corrected_pyrads_os_spear[i]=0.0005
pvals2_corrected_racat_os_spear = p_adjust_bh(pvals2_racat_os_spear)
for i, val in enumerate(pvals2_corrected_racat_os_spear):
	if val == 0.:
		pvals2_corrected_racat_os_spear[i]=0.0005

pvals1_corrected_pyrads_pfs_spear = p_adjust_bh(pvals1_pyrads_pfs_spear)
for i, val in enumerate(pvals1_corrected_pyrads_pfs_spear):
    if val == 0.:
        pvals1_corrected_pyrads_pfs_spear[i]=0.0005
pvals2_corrected_racat_pfs_spear = p_adjust_bh(pvals2_racat_pfs_spear)
for i, val in enumerate(pvals2_corrected_racat_pfs_spear):
    if val == 0.:
        pvals2_corrected_racat_pfs_spear[i]=0.0005


pvals1_corrected_pyrads_cd8_spear = p_adjust_bh(pvals1_pyrads_cd8_spear)
for i, val in enumerate(pvals1_corrected_pyrads_cd8_spear):
    if val == 0.:
        pvals1_corrected_pyrads_cd8_spear[i]=0.0005
pvals2_corrected_racat_cd8_spear = p_adjust_bh(pvals2_racat_cd8_spear)
for i, val in enumerate(pvals2_corrected_racat_cd8_spear):
    if val == 0.:
        pvals2_corrected_racat_cd8_spear[i]=0.0005


pvals1_corrected_pyrads_os_pear = p_adjust_bh(pvals1_pyrads_os_pear)
for i, val in enumerate(pvals1_corrected_pyrads_os_pear):
    if val == 0.:
        pvals1_corrected_pyrads_os_pear[i]=0.0005
pvals2_corrected_racat_os_pear = p_adjust_bh(pvals2_racat_os_pear)
for i, val in enumerate(pvals2_corrected_racat_os_pear):
    if val == 0.:
        pvals2_corrected_racat_os_pear[i]=0.0005

pvals1_corrected_pyrads_pfs_pear = p_adjust_bh(pvals1_pyrads_pfs_pear)
for i, val in enumerate(pvals1_corrected_pyrads_pfs_pear):
    if val == 0.:
        pvals1_corrected_pyrads_pfs_pear[i]=0.0005
pvals2_corrected_racat_pfs_pear = p_adjust_bh(pvals2_racat_pfs_pear)
for i, val in enumerate(pvals2_corrected_racat_pfs_pear):
    if val == 0.:
        pvals2_corrected_racat_pfs_pear[i]=0.0005


pvals1_corrected_pyrads_cd8_pear = p_adjust_bh(pvals1_pyrads_cd8_pear)
for i, val in enumerate(pvals1_corrected_pyrads_cd8_pear):
    if val == 0.:
        pvals1_corrected_pyrads_cd8_pear[i]=0.0005
pvals2_corrected_racat_cd8_pear = p_adjust_bh(pvals2_racat_cd8_pear)
for i, val in enumerate(pvals2_corrected_racat_cd8_pear):
    if val == 0.:
        pvals2_corrected_racat_cd8_pear[i]=0.0005



fdr1_pyrads_os_spear = [-1 * math.log10(p) for p in pvals1_pyrads_os_spear]
fdr2_racat_os_spear = [-1 * math.log10(p) for p in pvals2_racat_os_spear]

fdr1_pyrads_pfs_spear = [-1 * math.log10(p) for p in pvals1_pyrads_pfs_spear]
fdr2_racat_pfs_spear = [-1 * math.log10(p) for p in pvals2_racat_pfs_spear]


fdr1_pyrads_cd8_spear = [-1 * math.log10(p) for p in pvals1_pyrads_cd8_spear]
fdr2_racat_cd8_spear = [-1 * math.log10(p) for p in pvals2_racat_cd8_spear]

fdr1_pyrads_os_pear = [-1 * math.log10(p) for p in pvals1_pyrads_os_pear]
fdr2_racat_os_pear = [-1 * math.log10(p) for p in pvals2_racat_os_pear]

fdr1_pyrads_pfs_pear = [-1 * math.log10(p) for p in pvals1_pyrads_pfs_pear]
fdr2_racat_pfs_pear = [-1 * math.log10(p) for p in pvals2_racat_pfs_pear]

fdr1_pyrads_cd8_pear = [-1 * math.log10(p) for p in pvals1_pyrads_cd8_pear]
fdr2_racat_cd8_pear = [-1 * math.log10(p) for p in pvals2_racat_cd8_pear]


print(sum(pvals1_pyrads_os_pear<0.05))
print(sum(pvals1_pyrads_os_spear<0.05))

print(sum(pvals2_racat_os_pear<0.05))
print(sum(pvals2_racat_os_spear<0.05))

print(sum(pvals1_pyrads_pfs_pear<0.05))
print(sum(pvals1_pyrads_pfs_spear<0.05))

print(sum(pvals2_racat_pfs_pear<0.05))
print(sum(pvals2_racat_pfs_spear<0.05))



#dataframe of Pyrads correlation results
data_pyrads = {'corr_pyrads_spear_pfs_bw25':correlations_pyrads_pfs_spear[1:], 'pval_pyrads_spear_pfs_bw25':pvals1_pyrads_pfs_spear[1:],
'corr_pyrads_pear_pfs_bw25':correlations_pyrads_pfs_pear[1:], 'pval_pyrads_pear_pfs_bw25':pvals1_pyrads_pfs_pear[1:],
'corr_pyrads_spear_pfs_bw50':correlations_racat_pfs_spear[1:], 'pval_pyrads_spear_pfs_bw50':pvals2_racat_pfs_spear[1:],
'corr_pyrads_pear_pfs_bw50':correlations_racat_pfs_pear[1:], 'pval_pyrads_pear_pfs_bw50':pvals2_racat_pfs_pear[1:] ,

'corr_pyrads_spear_os_bw25':correlations_pyrads_os_spear[1:], 'pval_pyrads_spear_os_bw25':pvals1_pyrads_os_spear[1:],
'corr_pyrads_pear_os_bw25':correlations_pyrads_os_pear[1:], 'pval_pyrads_pear_os_bw25':pvals1_pyrads_os_pear[1:],
'corr_pyrads_spear_os_bw50':correlations_racat_os_spear[1:], 'pval_pyrads_spear_os_bw50':pvals2_racat_os_spear[1:],
'corr_pyrads_pear_os_bw50':correlations_racat_os_pear[1:], 'pval_pyrads_pear_os_bw50':pvals2_racat_os_pear[1:] }

df_pyrads = pd.DataFrame(data_pyrads)


#label the significant correlations for each evaluator
df_pyrads['pfs_spear_bw25'] = df_pyrads['pval_pyrads_spear_pfs_bw25'] < 0.05
df_pyrads['pfs_spear_bw50'] = df_pyrads['pval_pyrads_spear_pfs_bw50'] < 0.05
df_pyrads['pfs_pear_bw25'] = df_pyrads['pval_pyrads_pear_pfs_bw25'] < 0.05
df_pyrads['pfs_pear_bw50'] = df_pyrads['pval_pyrads_pear_pfs_bw50'] < 0.05

df_pyrads['os_spear_bw25'] = df_pyrads['pval_pyrads_spear_os_bw25'] < 0.05
df_pyrads['os_spear_bw50'] = df_pyrads['pval_pyrads_spear_os_bw50'] < 0.05
df_pyrads['os_pear_bw25'] = df_pyrads['pval_pyrads_pear_os_bw25'] < 0.05
df_pyrads['os_pear_bw50'] = df_pyrads['pval_pyrads_pear_os_bw50'] < 0.05




exit()

# Create a dictionary to map labels to colors
#colors = {'Shape': 'red', 'Intensity': 'orange', 'Texture': 'green'}
colors = {'Texture': 'green', 'Intensity': 'orange', "Statistics" : "black", 'Shape': 'red'}



# set up figure and axis
fig, axs = plt.subplots(2,2, constrained_layout = True)

for label in set(features[1:]):
    indices = [i for i, x in enumerate(features[1:]) if x == label]
    axs[0,0].scatter([fdr1_pyrads_os_pear[i] for i in indices], [fdr2_racat_os_pear[i] for i in indices], color=colors[label], label=label, s=25,alpha=0.6)
    axs[0,0].grid(True)
    axs[0,0].set_ylabel('bin count 16 ('+ r'$-log10(p_{Pearson}$)'+')', labelpad = 15, fontsize = 15)
    axs[0,0].set_xlabel('bin count 128 ('+ r'$-log10(p_{Pearson}$)'+')', labelpad = 15, fontsize = 15)
    axs[0,0].set_title('OS', fontsize = 15, y= 1.03)

    axs[0,0].set_ylim([-0.5, 4])
    axs[0,1].set_ylim([-0.5, 4])
    axs[0,0].set_xlim([-0.5, 4])
    axs[0,1].set_xlim([-0.5, 4])
    axs[1,0].set_ylim([-0.5, 5])
    axs[1,1].set_ylim([-0.5, 5])
    axs[1,0].set_xlim([-0.5, 5])
    axs[1,1].set_xlim([-0.5, 5])

    axs[0,1].scatter([fdr1_pyrads_pfs_pear[i] for i in indices], [fdr2_racat_pfs_pear[i] for i in indices], color=colors[label], label=label, s=25,alpha=0.6)
    axs[0,1].legend(fontsize = 15, loc="upper right")
    axs[0,1].grid(True)
    axs[0,1].set_title('PFS', fontsize = 15, y= 1.03)
    axs[0,1].set_xlabel('bin count 128 ('+ r'$-log10(p_{Pearson}$)'+')', labelpad = 15, fontsize = 15)
#    axs[0,2].scatter([fdr1_pyrads_cd8_pear[i] for i in indices], [fdr2_racat_cd8_pear[i] for i in indices], color=colors[label], label=label)    
    axs[1,0].scatter([fdr1_pyrads_os_spear[i] for i in indices], [fdr2_racat_os_spear[i] for i in indices], color=colors[label], label=label, s=25,alpha=0.6)
    axs[1,0].grid(True)
    axs[1,0].set_ylabel('bin count 16 ('+ r'$-log10(p_{Spearman}$)'+')', labelpad = 15, fontsize = 15)
    axs[1,0].set_xlabel('bin count 128 ('+ r'$-log10(p_{Spearman}$)'+')', labelpad = 15, fontsize = 15)    
    axs[1,1].scatter([fdr1_pyrads_pfs_spear[i] for i in indices], [fdr2_racat_pfs_spear[i] for i in indices], color=colors[label], label=label, s=25,alpha=0.6)
    axs[1,1].grid(True)
    axs[1,1].set_xlabel('bin count 128 ('+ r'$-log10(p_{Spearman}$)'+')', labelpad = 15, fontsize = 15)        
#    axs[1,2].scatter([fdr1_pyrads_cd8_spear[i] for i in indices], [fdr2_racat_cd8_spear[i] for i in indices], color=colors[label], label=label)
#plt.legend( fontsize = 15)
#plt.setp(axs, xlim=axs[0,0].set_xlim([-0.5, 5.0]))

plt.show()

exit()          
# set boxplot properties
ax1 = axs[0,0].boxplot(p_values_platform1_os_pear, positions=[1, 3, 5, 7], boxprops=boxprops, flierprops=flierprops)
boxprops = dict(linestyle='-', linewidth=2, color='orange')
ax2 = axs[0,0].boxplot(p_values_platform2_os_pear, positions=[2, 4, 6, 8], boxprops=boxprops, flierprops=flierprops)
#axs[0,0].legend([ax1["boxes"][0], ax2["boxes"][0]], ['Pyrads', 'RaCat'], loc='upper left')
# add axis labels and title
axs[0,0].set_xticks([1.5 , 3.5, 5.5, 7.5])
#axs[0,0].set_xticklabels(['Statistics', 'Shape', 'Intensity', 'Texture'])
axs[0,0].set_xticklabels([])
#axs[0,0].set_xlabel('Groups of radiomic features', labelpad = 15, fontsize = 15)
axs[0,0].set_ylabel('p-values (Pearson)', labelpad = 15, fontsize = 15)
axs[0,0].set_title('OS', fontsize = 15, y= 1.03)



# Plot the scatter points
for label in set(features[1:]):
    indices = [i for i, x in enumerate(features[1:]) if x == label]
    plt.scatter([fdr1[i] for i in indices], [fdr2[i] for i in indices], color=colors[label], label=label)

# Add axis labels and a legend
plt.xlabel('-log10(FDR) from Pyrads', fontsize = 15, labelpad=15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylabel('-log10(FDR) from RaCat', fontsize = 15, labelpad=15)
plt.legend(fontsize =15)
plt.title("Comparison of FDR from correlation of radiomics with OS", fontsize =20)

# Display the plot
#fig.tight_layout(pad=10.0)
plt.show()
