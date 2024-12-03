# PTP ML-Hybrid Ensemble Approach
Building off of the success of the SET8 ML-Hybrid Ensemble Method, and the generalisation of that method to SIRT1-7, this investigation of the protein tyrosine phosphatase family exhibits the generalizability of the technique.
Dr. Gianni Cesareni and Dr. Michele Tinti provided guidance regarding their dataset from the 2017 publication, "Both Intrinsic Substrate Preference and Network Context Contribute to Substrate Selection of Classical Tyrosine Phosphatases" which contained a dataset of 6,400 tyrosine-centric peptides representative of potential substrates within proteins tested with over 15 different phosphorylases. This extensive dataset will be applied to generate 15 different ML models to predict each corresponding phosphorylases' activity towards tyrosines across the proteome.

## 1_Phosphorylation_Feature_Generation.ipynb
This notebook further generalises the manual generation of features for the PML pipeline. Modified to accommodate for phosphorylation/kinase based datasets, this code enables the user to generate the MACCS key, ProtDCal, and one-hot encoding features required to run the hybrid method on peptide array data.

## 2a_MusiteDeep_Post_Processing.ipynb
Following [MusiteDeep](https://academic.oup.com/nar/article/48/W1/W140/5824154) scoring of the features, this notebook aids in the handling and formatting of the scores such that they may be incorporated into the ensemble model.

## 2b_Cleaning_of_Data_Based_on_Suggestions_from_Authors.ipynb
[Dr. Gianni Cesareni and Dr. Michele Tinti provided guidance regarding their dataset from the 2017 publication](https://pubmed.ncbi.nlm.nih.gov/28159843/), however only PTP1B and DEP-1 could be confidently handled. Hence, a static cutoff was applied to determine a peptide spot positive or negative for activity for the other 13 PTPs. Due to this, these additional investigations are incorporated within the supplementary files.

## 2c_Musite_Deep_Post_Run_Processing_for_Full_Tyr_Proteome.ipynb
Following MusiteDeep scoring of the tyrosine-centric sites of the proteome, this notebook aids in the handling and formatting of the scores such that the proteome may be scored by the ensemble model.

## Feature_Generation_for_Full_Tyr_Proteome.ipynb
This notebook initiates and handles the feature generation of the entire tyrosine proteome. 

## 2e_MusiteDeep_and_Feature_Generation_for_MS_Verified_Substrates.ipynb
Obtained from a [2011 publication by Ren et. al](https://pmc.ncbi.nlm.nih.gov/articles/PMC3074353/), a compiled set of sites for PTP1B, SHP1 and SHP2 were obtained from prior investigations. These lists were filtered to remove sites that were already present within the training data, leaving never-before-seen known sites of PTP activity to be scored by the respective models. The efficacy of the models in identifying these sites correctly informs our precision value, later reported within the main publication.

## 3_ML_Runs.ipynb
A looped notebook containing calls to the ML functions related to creating ML-hybrid ensemble models for all 15 PTPs, with included scoring of the experimental datasets for PTP1B, SHP1, and SHP2 as mentioned above. 

[Many of the files referenced by these notebooks were too large to be included on this page.](https://drive.google.com/drive/folders/1DhendsPklUZK6lIfDl2LEHOEpZSpoz1-?usp=share_link) Please upload this GitHub repository locally, along with the large files, such that the notebooks can function effectively.
