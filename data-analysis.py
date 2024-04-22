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
main_dir = Path('/media/sf_Shared-Linux/Shared-Linux/IUCPQ/EGFR_prediction/test_samples/iucpq-Venkata/')
#read the features from an Excel file
rad_data = pd.read_excel(os.path.join(main_dir, 'Results/DGX_output/radiomicsfeatures_pyrads_binwidth25.xlsx')) 

#racat_data = pd.read_csv(os.path.join(main_dir, 'Results/PC_output/RaCat_concat_bin25.csv')) 

#read the additinal data from .Rdata
addi_data = pyreadr.read_r(os.path.join(main_dir,"iucpq_data/oncotech_container_IUCPQ_clinical_ihc_RNAseq_pyrads_FREEZEmarch2022.Rdata"))

#read data from .Rdata file
clinical_data = pd.DataFrame(addi_data["clinical_IUCPQ"])
IHC_data = pd.DataFrame(addi_data["IHC_IUCPQ"])
rads_iucpq_data = pd.DataFrame(addi_data["pyrads_IUCPQ"])

radiomics_headers_iucpq_data = list(rads_iucpq_data.columns)

#find the index of the samples in radiomics data
index_rad_data = list(rad_data["Unnamed: 0"])

#find the index of samples in additional data (clinical+endpoints)
index_clinical_data = addi_data["clinical_IUCPQ"]["oncotech_id"]
index_IHC_data = addi_data["IHC_IUCPQ"]["oncotech_id"]
index_rads_iucpq_data = addi_data["pyrads_IUCPQ"]["oncotech_id"]

#set the labels as index for radiomics and each group of additinal data
rad_data = rad_data.set_index(pd.Series(index_rad_data))
clinical_data = clinical_data.set_index(index_clinical_data)
IHC_data = IHC_data.set_index(index_IHC_data)
rads_iucpq_data = rads_iucpq_data.set_index(index_rads_iucpq_data)

#find the common labels between radiomics data and each group of additional data
index_intersection_clinical = [value for value in index_rad_data if value in list(index_clinical_data)]
index_intersection_IHC = [value for value in index_rad_data if value in list(index_IHC_data)]
index_intersection_rads_iucpq = [value for value in index_rad_data if value in list(index_rads_iucpq_data)]

#select the additinal data with available radiomic features (from intersected labels)
clinical_data_intersected = clinical_data.loc[index_intersection_clinical]
IHC_data_intersected = IHC_data.loc[index_intersection_IHC]
rads_iucpq_data_intersected = rads_iucpq_data.loc[index_intersection_rads_iucpq]

#find the common labels between radiomics data and all the additional data
index_intersection_total = [value for value in index_rad_data if value in list(index_intersection_IHC) and value in list(index_intersection_clinical)]
#select the patients with available additinal and radiomic features (from total intersected labels)
clinical_data_intersected_total = clinical_data.loc[index_intersection_total]
IHC_data_intersected_total = IHC_data.loc[index_intersection_total]
rads_iucpq_data_intersected_total = rads_iucpq_data.loc[index_intersection_total]

# select the float columns from rad_data
rad_data = rad_data.select_dtypes(include=[float])
# examining missing values from rad_data
#print(reader_data.isnull().values.any())
#print(reader_data.isnull().sum().sum())

#place inf with nan in rad_data
rad_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#drop the columns which include missing values from the dataset
rad_data.dropna(inplace=True, axis = 1)

#remove all features that are constant 
rad_data = rad_data.loc[:, rad_data.var() != 0.0]
#print("Shape of the features after removing constant features =", np.shape(rad_data))
radiomics_headers_rads = list(rad_data.columns)
#find the features with same label between IUCPQ pyrads and my pyrads
radiomics_common_header = [value for value in radiomics_headers_rads if value in radiomics_headers_iucpq_data]
rads_iucpq_common_rad_data = rads_iucpq_data.loc[:, radiomics_common_header]
rad_data_common_rads_iucpq = rad_data.loc[:, radiomics_common_header]
 
#select the patients with available additinal and radiomic features from iucpq and from pyrads(from total intersected labels)
rad_data_intersected_total = rad_data.loc[index_intersection_total]
rad_data_intersected = rad_data.loc[index_intersection_clinical]
rad_iucpq_data_intersected = rads_iucpq_data.loc[index_intersection_clinical]
"""
#save the Rdata file as an excel file
writer = pd.ExcelWriter(os.path.join(main_dir,'Results/data_analysis/Pyrads/clinical_IHC_rads_iucpq.xlsx'), engine='xlsxwriter')
wb  = writer.book
clinical_data.to_excel(writer, sheet_name="clinical_data")    ## write into excel
IHC_data.to_excel(writer, sheet_name="IHC")    ## write into excel
rads_iucpq_data.to_excel(writer, sheet_name="radiomics_iucpq")    ## write into excel
wb.close()

#save the clinical and IHC data for the patients with available radiomic features (available CT and seg)
writer = pd.ExcelWriter(os.path.join(main_dir,'Results/data_analysis/Pyrads/clinical_IHC_individual_inter_with_rads_bc512.xlsx'), engine='xlsxwriter')
wb  = writer.book
clinical_data_intersected.to_excel(writer, sheet_name="clinical_data")    ## write into excel
IHC_data_intersected.to_excel(writer, sheet_name="IHC")    ## write into excel
rads_iucpq_data_intersected.to_excel(writer, sheet_name="radiomics_iucpq")    ## write into excel
rad_data_intersected.to_excel(writer, sheet_name="radiomics_pyrads_bc512")    ## write into excel

wb.close()

#save the data for the patients with available radiomic features (available CT and seg)
writer = pd.ExcelWriter(os.path.join(main_dir, 'Results/data_analysis/Pyrads/clinical_IHC_rads_bw50_IUCPQ_intersection.xlsx'), engine='xlsxwriter')
wb  = writer.book
clinical_data_intersected_total.to_excel(writer, sheet_name="clinical_data")    ## write into excel
IHC_data_intersected_total.to_excel(writer, sheet_name="IHC")    ## write into excel
rads_iucpq_data_intersected_total.to_excel(writer, sheet_name="radiomics_iucpq")    ## write into excel
rad_data_intersected_total.to_excel(writer, sheet_name="radiomics_pyrads_bw50")    ## write into excel

wb.close()
"""
endpoint_label = ["os_days", "pfs_days"]


for label in endpoint_label:
	corr_prs_total =[]
	corr_spr_total = []
	corr_prs_pvalue_total =[]
	corr_spr_pvalue_total = []

	trg = clinical_data_intersected[label]
	for ftr in rad_iucpq_data_intersected.columns[1:]:
#		print(label)
		corr_prs = scipy.stats.pearsonr(rad_iucpq_data_intersected[ftr], trg)    # Pearson's r
		corr_spr = scipy.stats.spearmanr(rad_iucpq_data_intersected[ftr], trg)   # Spearman's rho
		corr_prs_total.append(corr_prs[0])
		corr_prs_pvalue_total.append(corr_prs[1])
		corr_spr_total.append(corr_spr[0])
		corr_spr_pvalue_total.append(corr_spr[1])


	feature_importances = pd.Series(list(map(abs, corr_prs_total)), index = rad_iucpq_data_intersected.columns[1:])
	feature_importances.nlargest(len(rad_iucpq_data_intersected.columns[1:]))
	rank_features = feature_importances.rank(ascending=False)
	writer = pd.ExcelWriter(os.path.join(main_dir, 'Results/data_analysis/Pyrads/rank_iucpq_rads_bw25_'+label+'.xlsx'), engine='xlsxwriter')
	wb  = writer.book
	rank_features.to_excel(writer, sheet_name="rank_iucpq_rads_bw25_"+label)    ## write into excel 
	wb.close()

	writer = pd.ExcelWriter(os.path.join(main_dir, 'Results/data_analysis/Pyrads/correlation_iucpq_rads_bw25_'+label+'.xlsx'), engine='xlsxwriter')
	wb  = writer.book
	df = pd.DataFrame(corr_prs_total, columns = ["corr_prs"], index = rad_iucpq_data_intersected.columns[1:])     ## put into a dataframe format
	df["p-value_prs"] = corr_prs_pvalue_total
	df["corr_spr"] = corr_spr_total     ## put into a dataframe format
	df["p-value_spr"] = corr_spr_pvalue_total
	df.to_excel(writer, sheet_name="corr_rads_bw25_"+label)    ## write into excel 
	wb.close()
"""
	#Visualize the importance of features for F-test regression
	plt.figure(num=None, figsize=(120,100), dpi=200, facecolor='w', edgecolor='k')
	feature_importances = pd.Series(list(map(abs, corr_prs_total)), index = rad_data_intersected_total.columns)
	feature_importances.nlargest(len(rad_data_intersected_total.columns)).plot(kind='barh', color='lightsteelblue')
	plt.ylabel("Feature", fontsize= 80, labelpad= 50)
	plt.xlabel("Score (Pearson CC)", fontsize= 80, labelpad= 50)
	plt.xticks(fontsize=70)
	plt.yticks(fontsize=5)
	plt.title("The importance of the features for " +label+ " prediction" +" from Pyrads with bincount = 512", fontdict={'fontsize':80}, y= 1.02)
	plt.savefig(os.path.join(main_dir, "importance_cc_pear_Pyrads_bw50_"+label +"_.png"))

"""
exit()
#save the radiomics from iucpq data and the extracted radiomic features from pyrads with the same labels in one excel file
writer = pd.ExcelWriter(os.path.join(main_dir,'Results/data_analysis/Pyrads/radiomicsfeatures_pyrads_bw50_clean_common_with_iucpq_radiomics_labels.xlsx'), engine='xlsxwriter')
wb  = writer.book
rad_data_common_rads_iucpq.to_excel(writer, sheet_name="pyrads_me")    ## write into excel
rads_iucpq_common_rad_data.to_excel(writer, sheet_name="pyrads_iucpq_data")
wb.close()


#save the extracted radiomic features from pyrads or Racat after cleaning
writer = pd.ExcelWriter(os.path.join(main_dir,'Results/data_analysis/Pyrads/radiomicsfeatures_pyrads_bw50_clean.xlsx'), engine='xlsxwriter')
wb  = writer.book
rad_data.to_excel(writer, sheet_name="radiomic features")    ## write into excel
wb.close()





exit()
#get correlations between the features in dataset 
corr_matrix = rad_data.corr()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
# Find the features (columns) with correlation greater than 0.9 which may be dropped
column_to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
#print(len(column_to_drop))

#plot the heatmap of the correlation Matrix
corr_features_ind = corr_matrix.index
print(len(corr_features_ind))
plt.figure(figsize=(50,50))
ax=sns.heatmap(rad_data[corr_features_ind].corr(),annot=False,cmap="RdYlGn", vmin=-1,vmax=1)
# use matplotlib.colorbar.Colorbar object
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=40)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig(os.path.join(main_dir, "Results/Plots/RaCat_concat_bin25.png"), bbox_inches='tight')

#drop the features with high correlation
rad_data = rad_data.drop(labels = column_to_drop, axis=1)

#plot the heatmap of the correlation Matrix after dropping the most correlated features
corr_matrix_dropped = reader_data.corr()
corr_features_ind = corr_matrix_dropped.index
plt.figure(figsize=(50,50))
ax=sns.heatmap(reader_data[corr_features_ind].corr(),annot=False,cmap="RdYlGn", vmin=-1,vmax=1)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=40)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig(os.path.join(main_dir, "Results/Plots/RaCat_concat_bin25_dropped.png"), bbox_inches='tight')
