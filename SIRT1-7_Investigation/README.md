# SIRT1-7 ML-Hybrid_Ensemble
Building off of the success of the SET8 ML-Hybrid Ensemble Method, this investigation of the sirtuin family exhibits the generalizability of the technique. Fitting and scoring is now automated, set to return the model that best-performs on the dataset, as defined by a metric dependent on the data imbalance. 

## Feature Generation
Features for the training dataset, and the experimental dataset are generated as in the SET8 investigation. N6-acetyllysine prediction scores were obtained from the [MuSite Deep](https://academic.oup.com/nar/article/48/W1/W140/5824154) webserver for this training dataset, as well as a whole proteome set. 

## SIRT_ALL_RUNS
This notebook applies the same ML-Hybrid Ensemble Methodology as outlined within the SET8 investigation to all seven sirtuins (SIRTs) within the training dataset ([as obtained from this publication](https://www.nature.com/articles/ncomms3327)). To begin with a base ML model is best-fit to the training data, then data balancing methods (if dataset imbalance is greater than 20:80) and hyperparameter tuning are applied. Next, an ensemble model is tested and fit, through either soft-voting or stacking. Performance is assessed and the resulting combination is applied to score the same experimental surface exposed dataset as applied to the SET8 investigation. Outputs are model performances (upon the training + validation datasets for fit assessment), as well as feature importance of the base ML model, and a dataset containing the scored experimental proteome. 

## NOTE
Some datasets were too large to be uploaded to this page. They include the full proteome surface exposed lysine set, and the MethylSight full proteome set. If downloading locally, deposit in a separate file called "big_datasets". [Large files may be found here.](https://drive.google.com/drive/folders/16vLRi1fO6vgmI8lwu0RxCW_Vr8JOC6EO?usp=sharing)
