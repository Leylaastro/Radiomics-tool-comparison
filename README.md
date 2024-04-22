# Radiomics-tool-comparison
**Project Overview**
This project aims to investigate the correlation between radiomic features extracted using Pyradiomics and RaCat, and their association with immunotherapy biomarkers in patients with Non-Small Cell Lung Cancer (NSCLC). This analysis utilizes two different gray level discretization settings in Pyradiomics.

We utilize computed tomography (CT) annotated images of 164 patients, provided in an h5 file, to extract the relevant radiomic features.

**Scripts and Usage**
read_h5.py
Purpose: Reads the .h5 file containing patient data and exports essential identifiers (patient_ID, study_ID, and series_ID) to an Excel file.
Output: patient_details.xlsx

pyrads_extraction_h5.py
Purpose: Extracts radiomic features for 164 studies using Pyradiomics from the .h5 file.
Configuration: Requires a parameter file for feature customization (Params.yaml).
Output: pyradiomics_features.xlsx

RaCat-commandline.txt
Description: Contains the command line instructions to extract radiomics using RaCat.
Related Config File: RaCat_sample_config.ini (configure this file as per the requirements to use with RaCat).

**Configuration Files**
Params.yaml
Description: Provides parameters for Pyradiomics feature extraction. The path to this file must be specified in pyrads_extraction_h5.py.
RaCat_sample_config.ini
Description: Configuration file for RaCat, used to specify settings for radiomic extraction from annotated CT images. The path to this file must be specified in the command in RaCat-commandline.txt

**Additional Scripts**
Analysis Scripts: Located in analysis/, these Python files are used for further analysis of the data and for calculating correlations as discussed in our associated paper (currently under review).

**Data**
Due to privacy and data protection laws, patient data in .h5 format is not publicly available in this repository. 
