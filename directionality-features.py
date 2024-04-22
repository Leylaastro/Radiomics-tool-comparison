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
df_pyrads_pvals_pfs = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_common_features_pyrads_pfs_new.xlsx'))
df_racat_pvals_pfs = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_common_features_racat_pfs_new.xlsx'))

df_pyrads_pvals_cd8 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_common_features_pyrads_CD8_new.xlsx'))
df_racat_pvals_cd8 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_common_features_racat_CD8_new.xlsx'))

df_pyrads_pvals_pdl1 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_common_features_pyrads_pdl1_new.xlsx'))
df_racat_pvals_pdl1 = pd.read_excel(os.path.join(main_dir, 'Results/data_analysis/final-analysis/evaluators_common_features_racat_pdl1_new.xlsx'))

df_features = df_racat_pvals_pfs["features"]

group = ["shape"]*2
markers2 = ["intensity"]*10
markers3 = ["texture"]*62

group.extend(markers2)
group.extend(markers3)



ind_pfs_pyrads_spear_pos = [i for i, x in enumerate(df_pyrads_pvals_pfs['corr_pyrads_spear']) if x>=0]
ind_pfs_pyrads_spear_neg = [i for i, x in enumerate(df_pyrads_pvals_pfs['corr_pyrads_spear']) if x<0]
ind_pfs_pyrads_spear_sig = [i for i, x in enumerate(df_pyrads_pvals_pfs['pval_pyrads_spear']) if x<0.05]


features_pfs_pyrads_spear_pos = df_features[ind_pfs_pyrads_spear_pos]
features_pfs_pyrads_spear_neg = df_features[ind_pfs_pyrads_spear_neg]
#print(features_os_pyrads_spear)
#print(len(ind_pfs_pyrads_spear_pos))
#print(len(ind_pfs_pyrads_spear_neg))

ind_cd8_pyrads_spear_pos = [i for i, x in enumerate(df_pyrads_pvals_cd8['corr_pyrads_spear']) if x>=0]
ind_cd8_pyrads_spear_neg = [i for i, x in enumerate(df_pyrads_pvals_cd8['corr_pyrads_spear']) if x<0]
ind_cd8_pyrads_spear_sig = [i for i, x in enumerate(df_pyrads_pvals_cd8['pval_pyrads_spear']) if x<0.05]

features_cd8_pyrads_spear_pos = df_features[ind_cd8_pyrads_spear_pos]
features_cd8_pyrads_spear_neg = df_features[ind_cd8_pyrads_spear_neg]
#print(features_os_pyrads_spear)
#print(len(ind_cd8_pyrads_spear_pos))
#print(len(ind_cd8_pyrads_spear_neg))

ind_pdl1_pyrads_spear_pos = [i for i, x in enumerate(df_pyrads_pvals_pdl1['corr_pyrads_spear']) if x>=0]
ind_pdl1_pyrads_spear_neg = [i for i, x in enumerate(df_pyrads_pvals_pdl1['corr_pyrads_spear']) if x<0]
ind_pdl1_pyrads_spear_sig = [i for i, x in enumerate(df_pyrads_pvals_pdl1['pval_pyrads_spear']) if x<0.05]

features_pdl1_pyrads_spear_pos = df_features[ind_pdl1_pyrads_spear_pos]
features_pdl1_pyrads_spear_neg = df_features[ind_pdl1_pyrads_spear_neg]
#print(features_os_pyrads_spear)
#print(len(ind_pdl1_pyrads_spear_pos))
#print(len(ind_pdl1_pyrads_spear_neg))


ind_pfs_racat_spear_pos = [i for i, x in enumerate(df_racat_pvals_pfs['corr_racat_spear']) if x>=0]
ind_pfs_racat_spear_neg = [i for i, x in enumerate(df_racat_pvals_pfs['corr_racat_spear']) if x<0]
ind_pfs_racat_spear_sig = [i for i, x in enumerate(df_racat_pvals_pfs['pval_racat_spear']) if x<0.05]

features_pfs_racat_spear_pos = df_features[ind_pfs_racat_spear_pos]
features_pfs_racat_spear_neg = df_features[ind_pfs_racat_spear_neg]
#print(features_os_racat_spear)
#print(len(ind_pfs_racat_spear_pos))
#print(len(ind_pfs_racat_spear_neg))

ind_cd8_racat_spear_pos = [i for i, x in enumerate(df_racat_pvals_cd8['corr_racat_spear']) if x>=0]
ind_cd8_racat_spear_neg = [i for i, x in enumerate(df_racat_pvals_cd8['corr_racat_spear']) if x<0]
ind_cd8_racat_spear_sig = [i for i, x in enumerate(df_racat_pvals_cd8['pval_racat_spear']) if x<0.05]

features_cd8_racat_spear_pos = df_features[ind_cd8_racat_spear_pos]
features_cd8_racat_spear_neg = df_features[ind_cd8_racat_spear_neg]
#print(features_os_racat_spear)
#print(len(ind_cd8_racat_spear_pos))
#print(len(ind_cd8_racat_spear_neg))

ind_pdl1_racat_spear_pos = [i for i, x in enumerate(df_racat_pvals_pdl1['corr_racat_spear']) if x>=0]
ind_pdl1_racat_spear_neg = [i for i, x in enumerate(df_racat_pvals_pdl1['corr_racat_spear']) if x<0]
ind_pdl1_racat_spear_sig = [i for i, x in enumerate(df_racat_pvals_pdl1['pval_racat_spear']) if x<0.05]

features_pdl1_racat_spear_pos = df_features[ind_pdl1_racat_spear_pos]
features_pdl1_racat_spear_neg = df_features[ind_pdl1_racat_spear_neg]
#print(features_os_racat_spear)
#print(len(ind_pdl1_racat_spear_pos))
#print(len(ind_pdl1_racat_spear_neg))

pfs_pyrads_pos_racat_pos= list(set(ind_pfs_pyrads_spear_pos).intersection(ind_pfs_racat_spear_pos))
pfs_pyrads_neg_racat_neg= list(set(ind_pfs_pyrads_spear_neg).intersection(ind_pfs_racat_spear_neg))
pfs_pyrads_pos_racat_neg= list(set(ind_pfs_pyrads_spear_pos).intersection(ind_pfs_racat_spear_neg))
pfs_pyrads_neg_racat_pos= list(set(ind_pfs_pyrads_spear_neg).intersection(ind_pfs_racat_spear_pos))

pfs_pyrads_pos_racat_pos_pysig= list(set(pfs_pyrads_pos_racat_pos).intersection(ind_pfs_pyrads_spear_sig))
pfs_pyrads_pos_racat_pos_racsig= list(set(pfs_pyrads_pos_racat_pos).intersection(ind_pfs_racat_spear_sig))

pfs_pyrads_neg_racat_neg_pysig= list(set(pfs_pyrads_neg_racat_neg).intersection(ind_pfs_pyrads_spear_sig))
pfs_pyrads_neg_racat_neg_racsig= list(set(pfs_pyrads_neg_racat_neg).intersection(ind_pfs_racat_spear_sig))

pfs_pyrads_pos_racat_neg_pysig= list(set(pfs_pyrads_pos_racat_neg).intersection(ind_pfs_pyrads_spear_sig))
pfs_pyrads_pos_racat_neg_racsig= list(set(pfs_pyrads_pos_racat_neg).intersection(ind_pfs_racat_spear_sig))

pfs_pyrads_neg_racat_pos_pysig= list(set(pfs_pyrads_neg_racat_pos).intersection(ind_pfs_pyrads_spear_sig))
pfs_pyrads_neg_racat_pos_racsig= list(set(pfs_pyrads_neg_racat_pos).intersection(ind_pfs_racat_spear_sig))

cd8_pyrads_pos_racat_pos= list(set(ind_cd8_pyrads_spear_pos).intersection(ind_cd8_racat_spear_pos))
cd8_pyrads_neg_racat_neg= list(set(ind_cd8_pyrads_spear_neg).intersection(ind_cd8_racat_spear_neg))
cd8_pyrads_pos_racat_neg= list(set(ind_cd8_pyrads_spear_pos).intersection(ind_cd8_racat_spear_neg))
cd8_pyrads_neg_racat_pos= list(set(ind_cd8_pyrads_spear_neg).intersection(ind_cd8_racat_spear_pos))

cd8_pyrads_pos_racat_pos_pysig= list(set(cd8_pyrads_pos_racat_pos).intersection(ind_cd8_pyrads_spear_sig))
cd8_pyrads_pos_racat_pos_racsig= list(set(cd8_pyrads_pos_racat_pos).intersection(ind_cd8_racat_spear_sig))

cd8_pyrads_neg_racat_neg_pysig= list(set(cd8_pyrads_neg_racat_neg).intersection(ind_cd8_pyrads_spear_sig))
cd8_pyrads_neg_racat_neg_racsig= list(set(cd8_pyrads_neg_racat_neg).intersection(ind_cd8_racat_spear_sig))

cd8_pyrads_pos_racat_neg_pysig= list(set(cd8_pyrads_pos_racat_neg).intersection(ind_cd8_pyrads_spear_sig))
cd8_pyrads_pos_racat_neg_racsig= list(set(cd8_pyrads_pos_racat_neg).intersection(ind_cd8_racat_spear_sig))

cd8_pyrads_neg_racat_pos_pysig= list(set(cd8_pyrads_neg_racat_pos).intersection(ind_cd8_pyrads_spear_sig))
cd8_pyrads_neg_racat_pos_racsig= list(set(cd8_pyrads_neg_racat_pos).intersection(ind_cd8_racat_spear_sig))

pdl1_pyrads_pos_racat_pos= list(set(ind_pdl1_pyrads_spear_pos).intersection(ind_pdl1_racat_spear_pos))
pdl1_pyrads_neg_racat_neg= list(set(ind_pdl1_pyrads_spear_neg).intersection(ind_pdl1_racat_spear_neg))
pdl1_pyrads_pos_racat_neg= list(set(ind_pdl1_pyrads_spear_pos).intersection(ind_pdl1_racat_spear_neg))
pdl1_pyrads_neg_racat_pos= list(set(ind_pdl1_pyrads_spear_neg).intersection(ind_pdl1_racat_spear_pos))

pdl1_pyrads_pos_racat_pos_pysig= list(set(pdl1_pyrads_pos_racat_pos).intersection(ind_pdl1_pyrads_spear_sig))
pdl1_pyrads_pos_racat_pos_racsig= list(set(pdl1_pyrads_pos_racat_pos).intersection(ind_pdl1_racat_spear_sig))

pdl1_pyrads_neg_racat_neg_pysig= list(set(pdl1_pyrads_neg_racat_neg).intersection(ind_pdl1_pyrads_spear_sig))
pdl1_pyrads_neg_racat_neg_racsig= list(set(pdl1_pyrads_neg_racat_neg).intersection(ind_pdl1_racat_spear_sig))

pdl1_pyrads_pos_racat_neg_pysig= list(set(pdl1_pyrads_pos_racat_neg).intersection(ind_pdl1_pyrads_spear_sig))
pdl1_pyrads_pos_racat_neg_racsig= list(set(pdl1_pyrads_pos_racat_neg).intersection(ind_pdl1_racat_spear_sig))

pdl1_pyrads_neg_racat_pos_pysig= list(set(pdl1_pyrads_neg_racat_pos).intersection(ind_pdl1_pyrads_spear_sig))
pdl1_pyrads_neg_racat_pos_racsig= list(set(pdl1_pyrads_neg_racat_pos).intersection(ind_pdl1_racat_spear_sig))



print("PFS")
print(pfs_pyrads_pos_racat_pos)
print([group[i] for i in pfs_pyrads_pos_racat_pos])
print([df_features[i] for i in pfs_pyrads_pos_racat_pos])
print(pfs_pyrads_neg_racat_neg)
print([group[i] for i in pfs_pyrads_neg_racat_neg])
print([df_features[i] for i in pfs_pyrads_neg_racat_neg])

print([df_features[i] for i in pfs_pyrads_pos_racat_pos_pysig])
print([df_features[i] for i in pfs_pyrads_neg_racat_neg_pysig])

print([df_features[i] for i in pfs_pyrads_pos_racat_pos_racsig])
print([df_features[i] for i in pfs_pyrads_neg_racat_neg_racsig])

print(pfs_pyrads_pos_racat_neg)
print(pfs_pyrads_neg_racat_pos)

print(len(pfs_pyrads_pos_racat_pos)+len(pfs_pyrads_neg_racat_neg))
print(len(pfs_pyrads_pos_racat_pos_pysig)+len(pfs_pyrads_neg_racat_neg_pysig))
print(len(pfs_pyrads_pos_racat_pos_racsig)+len(pfs_pyrads_neg_racat_neg_racsig))

print(len(pfs_pyrads_pos_racat_neg)+len(pfs_pyrads_neg_racat_pos))
print(len(pfs_pyrads_pos_racat_neg_pysig)+len(pfs_pyrads_neg_racat_pos_pysig))
print(len(pfs_pyrads_pos_racat_neg_racsig)+len(pfs_pyrads_neg_racat_pos_racsig))


print("Pyrads CD8")

print(cd8_pyrads_pos_racat_pos)
print([group[i] for i in cd8_pyrads_pos_racat_pos])
print([df_features[i] for i in cd8_pyrads_pos_racat_pos])
print(cd8_pyrads_neg_racat_neg)
print([group[i] for i in cd8_pyrads_neg_racat_neg])
print([df_features[i] for i in cd8_pyrads_neg_racat_neg])
print(cd8_pyrads_pos_racat_neg)
print(cd8_pyrads_neg_racat_pos)

print(len(cd8_pyrads_pos_racat_pos)+len(cd8_pyrads_neg_racat_neg))
print(len(cd8_pyrads_pos_racat_pos_pysig)+len(cd8_pyrads_neg_racat_neg_pysig))
print(len(cd8_pyrads_pos_racat_pos_racsig)+len(cd8_pyrads_neg_racat_neg_racsig))

print(len(cd8_pyrads_pos_racat_neg)+len(cd8_pyrads_neg_racat_pos))
print(len(cd8_pyrads_pos_racat_neg_pysig)+len(cd8_pyrads_neg_racat_pos_pysig))
print(len(cd8_pyrads_pos_racat_neg_racsig)+len(cd8_pyrads_neg_racat_pos_racsig))




print("Pyrads PD-L1")

print([df_features[i] for i in pdl1_pyrads_pos_racat_pos_racsig])
print([df_features[i] for i in pdl1_pyrads_neg_racat_neg_racsig])
exit()
print(pdl1_pyrads_pos_racat_pos)
print([group[i] for i in pdl1_pyrads_pos_racat_pos])
print([df_features[i] for i in pdl1_pyrads_pos_racat_pos])
print(pdl1_pyrads_neg_racat_neg)
print([group[i] for i in pdl1_pyrads_neg_racat_neg])
print([df_features[i] for i in pdl1_pyrads_neg_racat_neg])
print(pdl1_pyrads_pos_racat_neg)
print(pdl1_pyrads_neg_racat_pos)

print(len(pdl1_pyrads_pos_racat_pos)+len(pdl1_pyrads_neg_racat_neg))
print(len(pdl1_pyrads_pos_racat_pos_pysig)+len(pdl1_pyrads_neg_racat_neg_pysig))
print(len(pdl1_pyrads_pos_racat_pos_racsig)+len(pdl1_pyrads_neg_racat_neg_racsig))

print(len(pdl1_pyrads_pos_racat_neg)+len(pdl1_pyrads_neg_racat_pos))
print(len(pdl1_pyrads_pos_racat_neg_pysig)+len(pdl1_pyrads_neg_racat_pos_pysig))
print(len(pdl1_pyrads_pos_racat_neg_racsig)+len(pdl1_pyrads_neg_racat_pos_racsig))



exit()
fig = plt.figure(figsize=(150,70))
#sns.scatterplot(x="Arc", y=0.5, hue="Category", data=df_MIc2_NOV_TRUE, s=100, marker="X")

#sns.scatterplot(x = "ind", y = 'corr_pyrads_spear',  data=df_pyrads_pvals_pfs, hue='group', s =100)
#sns.scatterplot(x = "ind", y = 'corr_racat_spear',  data=df_racat_pvals_pfs, hue='group', marker= "X", s =100)
plt.scatter(np.arange(1, len(df_pyrads_pvals_pfs['corr_pyrads_spear'])+1), df_pyrads_pvals_pfs['corr_pyrads_spear'], s=100, label = "Pyradiomics")
plt.scatter(np.arange(1, len(df_racat_pvals_pfs['corr_racat_spear'])+1), df_racat_pvals_pfs['corr_racat_spear'], s=100, label= "RaCat")

markers = ["x"]*2
markers2 = ["+"]*10
markers3 = ["o"]*62
markers.extend(markers2)
markers.extend(markers3)

#for xp, yp, m in zip(x, y1, markers):
#   plt.scatter(xp, yp, marker=m, s=100)

#for xp, yp, m in zip(x, y2, markers):
#   plt.scatter(xp, yp, marker=m, s=100)

plt.grid(ls="--")
plt.axhline(y=0.0, color='r', linestyle='-')
plt.ylabel( r'$\rho_{Spearman}$', fontsize= 30, labelpad=15)
plt.xlabel('feature', fontsize= 30, labelpad= 15)
plt.xticks(np.arange(0, len(df_pyrads_pvals_pfs['corr_pyrads_spear'])+1, 1), rotation = 90, color="w")
plt.yticks(fontsize = 20)
plt.legend(fontsize=20, loc ="upper left")
plt.show()


#


#plt.xticks(np.arange(0, len(arc_index)+1, 5))
#plt.yticks(np.arange(0.0, 80, 5))
#plt.savefig(os.path.join(main_dir_path, "NOV+TRUE_MI2_categorical.png"))