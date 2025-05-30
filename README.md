# ML-Hybrid Ensemble Method
Applying machine learning (ML) to the prediction of substrates for post-translational modification (PTM) inducing enzymes. 

We began our investigation with SET8, a lysine methyltransferase that affects both histone and non-histone substrates. The exploratory techniques applied may be observed within the SET8-Investigation directory. The datasets, and notebooks containing the methodology applied are outlined therein. The procedures applied yielded a significant rate of experimentally-validated precision for the identification of novel SET8 substrates, as validated by mass-spectrometry in human cells.

The basic workflow of the technique is as follows:
1. Generate data via high-throughput peptide array experiments using a directed subset of the proteome enriched for the PTM studied
2. Vectorize peptide data through features ([ProtDCal](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0586-0), [MACCS Keys](https://www.rdkit.org), and one-hot encoding of peptide sequence)
3. Fit a base ML model, apply data balancing method (if training data is imbalanced), and tune hyperparameters - assess resulting model
4. Fit an ensemble ML model through the use of a secondary, generalized PTM predictor (i.e. predicts if a PTM will take, not the enzyme responsible) - assess resulting model
5. Score an experimental dataset representative of the proteome to determine novel sites of PTM activity specific to the enzyme studied

To demonstrate the generalizability of the ML-Hybrid ensemble method towards other PTMs, the activity of the entire sirtuin family was predicted across the proteome, and validated with mass-spectrometry verified data for SIRT2. Additionally, the method was further applied towards the kinase Abl, a protein-tyrosine kinase. Much like the SIRT2 data, the Abl investigation is validated with mass-spectrometry verified data containing substrates that are highly likely to be modified by the enzyme. The procedures of the ML-Hybrid Ensemble Methodology were streamlined and automated for this process, as exhibited within the SIRT-Investigation directory. 

### Developed and Tested With python 3.11.4

### License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).  
You are free to use, share, and adapt the material for non-commercial purposes, as long as appropriate credit is given.
