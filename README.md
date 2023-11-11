## ML-Hybrid_Ensemble_SET8
This repository contains several descriptive notebooks that walk through the creation of the ML-Hybrid Ensemble Model. 

The ML-Hybrid Ensemble Model works off of a pregenerated experimental subset of the proteome, the methylome. The ~4,500 sites contained within the methylome were synthesised with peptide array experiments, then exposed to SET8. Sites that demonstrated SET8 methylation activity (~200) are noted within the file, and are our positive case. 

These sites are used to generate and train a base ML model, utilising the features described in the 1_Feature_Set_Generation.ipynb notebook within this repository. These generated features are then used within the 2_ML_Model_Fitting.ipynb notebook (also within this repository), to train and test the fit of several ML models, including the exploration of ensemble methods including voting and stacking with a secondary predictor, MethylSight (MS in file notation). Proceeding with a stacked ensemble model, the notebook 3_ML_Proteome_Scoring.ipynb within this repository scores a dataset using the ensemble model, be it the full proteome, a specified subset, or a proteome with missense mutations from cancer.


# 1_Feature_Set_Generation
In order to insert our peptide sequences into a ML model, we must first generate features. 

# 2_ML_Model_Fitting
This notebook outlines the procedures in fitting and testing models, including the ensemble models that apply MethylSight as the secondary ML predictor. Graphical representations of threshold v. metrics are included here as well, which is how a threshold that deems a prediction positive/negative is determined. 

# 3_ML_Proteome_Scoring
The final step is to re-initialize the model and apply it to score experimental datasets, as outlined here. It is important to note that the cancerous mutation datasets required initial runs with MethylSight, as the mutated sequences are not included in the full proteome file which concerns normal cells. 

# NOTE
Some datasets were too large to be uploaded to this page. They include the full proteome surface exposed lysine set, and the MethylSight full proteome set. They may be found here: https://drive.google.com/drive/folders/1mEmag5tNmzdm5_NIwxH43xEfKT-4aWi0?usp=share_link. 
