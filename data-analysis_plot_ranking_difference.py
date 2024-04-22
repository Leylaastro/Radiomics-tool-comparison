import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import os
from pathlib import Path
import pyreadr
import scipy

#path of the main folder
main_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ/EGFR_prediction/test_samples/iucpq-Venkata/Results/data_analysis/Pyrads/')

#read the features from an Excel file
rank_gr_1 = pd.read_excel(os.path.join(main_dir, 'bw25/ranking/rank_rads_bw25_pfs_days.xlsx')) 
rank_gr_2 = pd.read_excel(os.path.join(main_dir, 'bw50/ranking/rank_rads_bw50_pfs_days.xlsx')) 

index = rank_gr_1['Unnamed: 0']
index_sort = np.arange(len(index))
rank_gr_1_value = rank_gr_1[0]
rank_gr_2_value = rank_gr_2[0]
group2 = rank_gr_1 ['group2']
group3 = rank_gr_1 ['group3']
#print(group)
dif_rank = rank_gr_1_value- rank_gr_2_value
#dif_rank_series = pd.Series(list(map(abs, dif_rank)), index = group.tolist())
df_dif_rank = pd.DataFrame({'features': [str(val) for val in index_sort], 'features_name':index, 'ranking difference':list(map(abs, dif_rank)), 'class1' : group2, 'class2' : group3 })
#df_dif_rank_sorted = df_dif_rank.sort_values(by = ['dif_rank'])
#print(df_dif_rank_sorted['dif_rank'])


shape_rank = []
intensity_rank = []
textural_rank = []
for ind, val in enumerate(df_dif_rank['ranking difference']):
	if df_dif_rank['class2'][ind] == 'Shape':
		shape_rank.append(val)
	if df_dif_rank['class2'][ind] == 'Intensity':
		intensity_rank.append(val)
	if df_dif_rank['class2'][ind] == 'Textural':
		textural_rank.append(val)



#dif_rank_series_sorted = dif_rank_series.nsmallest(len(index))
#group = dif_rank_series_sorted.index

#data = {'radiomic features': index, 'the ranking difference': df_dif_rank_sorted['dif_rank'], 'class':df_dif_rank_sorted['group']}
plt.figure(figsize=(150,70))
ax = sns.barplot(x = 'features', y = 'ranking difference', data = df_dif_rank, hue = 'class2', order = df_dif_rank.sort_values(by = ['ranking difference']).features, width=2, palette=["red", "black", "cyan"])
#plt.legend(('Shape   '+str(min(shape_rank))+'-'+ str(max(shape_rank)), 'Intensity   '+str(min(intensity_rank))+'-'+ str(max(intensity_rank)),'Texture   '+str(min(textural_rank))+'-'+ str(max(textural_rank))), fontsize = 100, loc = 'upper left')


labels=['Shape-based       '+str(int(min(shape_rank)))+'-'+ str(int(max(shape_rank))), 'Intensity-based   '+str(int(min(intensity_rank)))+'-'+ str(int(max(intensity_rank))),'Texture-based     '+str(int(min(textural_rank)))+'-'+ str(int(max(textural_rank)))]

h, l = ax.get_legend_handles_labels()
ax.legend(h, labels, fontsize = 200, loc = 'upper left')
ax.set_xticklabels([])
#plt.legend(fontsize = 100, loc = 'upper left')
plt.ylabel("rank difference", fontsize= 200, labelpad= 200)
plt.xlabel("feature", fontsize= 200, labelpad= 200)
plt.xticks(fontsize=8, ticks = np.arange(1, len(index)+1), labels = None, rotation = 90)
plt.yticks(ticks = np.arange(0, max(df_dif_rank['ranking difference']), 150), fontsize=150)
plt.title("The difference of radiomic features rankings for PFS prediction between bin width = 25 and bin width = 50 results", fontdict={'fontsize':150}, y= 1.04)
ax.tick_params(axis='y', direction='out', length=30, width=10)
plt.savefig(os.path.join(main_dir, "ranking_dif_PFS_days_bw25_bw50.png"), bbox_inches='tight')




exit()
palette=["yellow", "red", "black", "cyan"]




#dif_rank = [abs(ele) for ele in dif_rank]
"""
fig, ax = plt.subplots(figsize=(10,6), facecolor=(.94, .94, .94))
bar = ax.bar(index, dif_rank)
plt.tight_layout()
plt.show()
"""




#dif_rank_series = pd.Series(list(map(abs, dif_rank)), index = index)
#dif_rank_series_sorted = dif_rank_series.nsmallest(len(index))



#dif_rank_series.plot(kind='bar', color='lightsteelblue')
#plt.ylabel("the rank difference", fontsize= 100, labelpad= 200)
#plt.xlabel("feature", fontsize= 100, labelpad= 200)
#plt.xticks(fontsize=8, rotation = 45)
#plt.yticks(fontsize=80)
#plt.title("The difference of radiomic features rankings within bin width = 25 and bin width = 50 results", fontdict={'fontsize':150}, y= 1.02)
#plt.savefig(os.path.join(main_dir, "ranking_dif_os_days_bw25_bw50.png"))


