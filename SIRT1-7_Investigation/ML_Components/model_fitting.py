# GENERAL LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# FOR ML
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import auc, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from numpy import mean
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import clone # NEW!!!!!
from sklearn.inspection import permutation_importance # NEW!!!!!
# ml models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import ComplementNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
# sampling methods
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek
from scipy.stats import loguniform


# K FOLD CROSS VALIDATION
def k_fold(X, y, model, sampling, sampling_type, scoring_method):
    
    ylist = y['EXPERIMENTALLY_ACTIVE'].tolist()
    
    # K Fold cross validation for model fitting - returns f1 score as metric
    if type(model) is str:
        scores = []
        print('k_fold: simp_avg in model name')
        if model == 'simp_avg_eq':
            pred_y = (X['PRIMARY_ML_SCORE'] + X['SECONDARY_ML_SCORE'])/2
        elif model == 'simp_avg_pr':
            pred_y = ((X['PRIMARY_ML_SCORE']*2) + X['SECONDARY_ML_SCORE'])/3
        else:
            pred_y = (X['PRIMARY_ML_SCORE'] + (X['SECONDARY_ML_SCORE']*2))/3
            
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        for i, (train_index, test_index) in enumerate(rskf.split(pred_y, y)):
            pred_y_binary = pred_y.iloc[test_index].tolist()
            pred_y_binary = [0 if num < 0.5 else 1 for num in pred_y_binary]
            if scoring_method == 'f1':
                scores.append(f1_score(y.iloc[test_index], pred_y_binary))
            else:
                scores.append(roc_auc_score(y.iloc[test_index], pred_y_binary))
    else:
        if sampling != 'none':
            steps = [(sampling_type, sampling), ('model', model)]
            pipeline = Pipeline(steps=steps)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(pipeline, X, ylist, cv=cv, n_jobs=1, scoring=scoring_method)
        else :
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            scores = cross_val_score(model, X,ylist, cv=cv, n_jobs=1, scoring=scoring_method)
    score = mean(scores)
    
    return score
    
# MODEL FITTING
def model_fitting_process(train_x, train_y, auto_fit, base_fitting, i_u_mod, i_u_samp, scoring_method):
    # train_x, train_y = training data for our model
    # auto_fit = T/F for whether or not the user wants to select their own model/sampling
    #  or for it to be automatically generated
    # test_balancing = T/F for whether or not we want to mess around with data balancinc
    
    # CREATE MODELS + SAMPLING METHODS FOR LATER USE
    # create our models
    dummy = DummyClassifier(strategy='most_frequent')
    logit = LogisticRegression(max_iter=1000)
    lda = LinearDiscriminantAnalysis()
    nb = ComplementNB()
    dtree = tree.DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    svc = svm.SVC()
    bagged = BaggingClassifier()
    rand_forest = RandomForestClassifier()
    ext_trees = ExtraTreesClassifier()
    gboost = GradientBoostingClassifier()

    # create list of our models to loop through -> replaced nb with dummy
    models = [dummy, logit, lda, dtree, knn, svc, bagged, rand_forest, ext_trees, gboost]

    # create our sampling methods
    # oversamplers
    over = RandomOverSampler()
    smote = SMOTE()
    border_smote = BorderlineSMOTE()
    svm_smote = SVMSMOTE()
    km_smote = KMeansSMOTE()
    adasyn = ADASYN()
    # undersamplers
    under = RandomUnderSampler()
#    cnn = CondensedNearestNeighbour() -> takes so long!
    tomek = TomekLinks()
    enn = EditedNearestNeighbours()
    n_cleaning = NeighbourhoodCleaningRule()
    onesided = OneSidedSelection()
    # combined samplers
    smoteenn = SMOTEENN()
    smotetomek = SMOTETomek()
    # none
    none = "none"
    
    # DETERMINE TRAINING DATA RATIO - IF OVER 0.2, DO NOT USE BALANCING METHODS
    # create list of sampling methods to loop through
    ratio = Counter(train_y.iloc[:,0])[1]/(len(train_y))
    print(ratio)
    if ratio < 0.2 or ratio > 0.8:
        samplings = [none, over, smote, border_smote, svm_smote, km_smote, adasyn, under, tomek, enn, n_cleaning, onesided, smoteenn, smotetomek]
        print('Data is imbalanced (', round(ratio*100, 2), '% pos) applying various sampling methods to remedy the issue...')
        if ratio < 0.05 or ratio > 0.95:
            models = [dummy, logit, lda, dtree, knn, svc, bagged, rand_forest, ext_trees, gboost]
        else:
            models = [dummy, logit, lda, knn, svc, bagged, rand_forest, ext_trees, gboost]
            
    else:
        samplings = [none]
        print('Data is balanced', round(ratio*100, 2), '% pos) will not be applying sampling methods, moving on...')
    
    
    if auto_fit == True:
        # NOW RUN THROUGH THE MODELS + SAMPLING TYPES TO DETERMINE THE BEST F-METRIC
        print('Now onto the automatic model fitting...')   # db

        ## FIT VARIOUS MODELS AND SAMPLING METHODS - SELECT FOR THE BEST ONE :) ##

        f1 = []   # scoring metric for the models
        
        if base_fitting == False:
            # We're in meta-learning mode- keep it simple and just run logistic regression
            #  and simple averaging
            models = [logit, 'simp_avg_eq', 'simp_avg_pr', 'simp_avg_sec']
            print('Meta model fitting commencing...')
        
        # Loop through our models
        for m in models:
            try:
                print('Model fitting of', m)   # db
                metric = k_fold(train_x, train_y, m, 'none', 'none', scoring_method)
            except:
                print('Model fitting of', m, 'failed, trying other models...')   # db
                metric = 0
            print('Model fitting of', m, 'resulted in an f-score of', metric)
            f1.append(metric)

        # Proceed with the model with the biggest f1 score
        model_max_f1 = max(f1)
        model_max = models[f1.index(model_max_f1)]

        # Secondly we loop through the balancing methods
        f1_score = 0
        f1_s = []   # scoring metric for the sampling method
        print('Now onto the balancing methods...')   # db
        
        if base_fitting == False:
            # If we're running meta learning we don't want to mess with balancing
            f1_s.append(model_max_f1)
            bal_max = "none"
        else:
            # Check sampling methods for best fit
            for s in samplings:
                if s == "none":
                    metric = model_max_f1
                else:
                    try:
                        print('Basic balancing testing of', s)   # db
                        metric = k_fold(train_x, train_y, model_max, s, 'sampling', scoring_method)
                    except:
                        print('Basic balancing testing of', s, 'failed, trying other methods...')   # db
                        metric = 0
                print('Balancing testing of', s, 'resulted in an f-score of', metric)
                f1_s.append(metric)

        # Proceed with the balancing method with the biggest f1 score
        bal_max_f1 = max(f1_s)
        bal_max = samplings[f1_s.index(bal_max_f1)]

        if bal_max == "none":
            bal_max = "none"

        # Return final f1 score
        f1_score = bal_max_f1

        # Return indices of model and sampling method that gave the highest f1 score
        model_max_index = f1.index(model_max_f1)
        bal_max_index = f1_s.index(bal_max_f1)
    
    else:
        # USER SELECTED THEIR OWN MODEL + SAMPLING COMBINATION
        # user-defined pipeline
        if samplings[i_u_samp] == "none":
            bal_max = "none"
            model_max = models[i_u_mod]
            f1_score = k_fold(train_x, train_y, model_max, 'none', 'none', scoring_method)
        else:
            model_max = models[i_u_mod]
            bal_max = samplings[i_u_samp]
            f1_score = k_fold(train_x, train_y, model_max, bal_max, 'sampling', scoring_method)
            
        print('User selected model', model_max, 'and balancing method', bal_max, 'resulted in an F-score of', f1_score)   # db
        
        # Set indices as those selected by the user
        model_max_index = i_u_mod
        bal_max_index = i_u_samp
    
    return model_max, bal_max, f1_score, model_max_index, bal_max_index
    
    
# GENERATE METRICS
def model_metric_generation(X_train, y_train, X_test, y_test, final_model, sampler):
    
    threshold = 0.0
    rec, prec, spec, sens, thresh, fpr, tpr = [], [], [], [], [], [], []
    
    if sampler != "none":
        # Apply sampling method to training data
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    
    if final_model == 'simp_avg_eq':
        y_proba = (X_test['PRIMARY_ML_SCORE'] + X_test['SECONDARY_ML_SCORE'])/2
        print('Simple average equal selected...')
    elif final_model == 'simp_avg_pr':
        y_proba = ((X_test['PRIMARY_ML_SCORE']*2) + X_test['SECONDARY_ML_SCORE'])/3
        print('Simple average primary weighted selected...')
    elif final_model == 'simp_avg_sec':
        y_proba = (X_test['PRIMARY_ML_SCORE'] + (X_test['SECONDARY_ML_SCORE']*2))/3
        print('Simple average secondary weighted selected...')
    else:
        # Fit the model to our training data
        final_model = clone(final_model)
        final_model = final_model.fit(X_train, y_train)

        # Calculate the probability on testing data
        y_proba = final_model.predict_proba(X_test)
        y_proba = y_proba[:, [1]]   #select the probability for the positive case only
    
    # Now generate metrics for each threshold from 0.0 - 1.0
    while threshold <= 1.01:
        # Select only the scores over our threshold
        cond = y_proba >= threshold
        
        # Convert to 1 or 0 values
        y_pred = np.where((y_proba>=threshold), 1, y_proba)
        y_pred = np.where((y_proba<threshold), 0, y_pred)
        
        # Calculate recall
        recall = recall_score(y_test, y_pred)
        rec.append(recall)
        
        # Calculate precision
        precision = precision_score(y_test, y_pred)
        prec.append(precision)
        
        #Calculate specificity + sensitivity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn/(tn+fp)
        spec.append(specificity)
        sensitivity = tp/(tp+fn)
        sens.append(sensitivity)
        
        # Calculate fpr + tpr
        false_pos_rate = fp/(fp+tn)
        fpr.append(false_pos_rate)
        true_pos_rate = tp/(tp+fn)
        tpr.append(true_pos_rate)
        
        # Update threshold
        thresh.append(threshold)
        threshold += 0.01
        
    # Determine ROC auc
    auc_m = roc_auc_score(y_test, y_proba)
    
    # Make metric dataframe
    pr_m = pd.DataFrame(list(zip(thresh, rec, spec, prec, sens)), columns=['Threshold'
                                                                              , 'Recall'
                                                                              , 'Specificity'
                                                                              , 'Precision'
                                                                              , 'Sensitivity'])

    # Make ROC dataframe
    roc_m = pd.DataFrame(list(zip(fpr, tpr)), columns = ['fpr', 'tpr'])
    
    return pr_m, roc_m, auc_m
    
# PRE-SCORED METRIC GENERATION
def pre_scored_metric_generation(y_test, y_proba):
    # Modified metric generation method for prescored values (i.e. if we want to generate metrics for graphing with our SECONDARY_ML_SCORE data

    # Now generate metrics for each threshold from 0.0 - 1.0
    threshold = 0.0
    rec, prec, spec, sens, thresh, fpr, tpr = [], [], [], [], [], [], []
    while threshold <= 1:
        # Select only the scores over our threshold
        cond = y_proba >= threshold

        # Convert to 1 or 0 values
        y_pred = np.where((y_proba>=threshold), 1, y_proba)
        y_pred = np.where((y_proba<threshold), 0, y_pred)

        # Calculate recall
        recall = recall_score(y_test, y_pred)
        rec.append(recall)

        # Calculate precision
        precision = precision_score(y_test, y_pred)
        prec.append(precision)

        #Calculate specificity + sensitivity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn/(tn+fp)
        spec.append(specificity)
        sensitivity = tp/(tp+fn)
        sens.append(sensitivity)

        # Calculate fpr + tpr
        false_pos_rate = fp/(fp+tn)
        fpr.append(false_pos_rate)
        true_pos_rate = tp/(tp+fn)
        tpr.append(true_pos_rate)

        # Update threshold
        thresh.append(threshold)
        threshold += 0.01

    # Determine ROC auc
    msd_roc_auc = roc_auc_score(y_test, y_proba)

    # Make metric dataframe
    msd_pr = pd.DataFrame(list(zip(thresh, rec, spec, prec, sens)), columns=['Threshold'
                                                                              , 'Recall'
                                                                              , 'Specificity'
                                                                              , 'Precision'
                                                                              , 'Sensitivity'])

    msd_pr['F-score'] = 2*((msd_pr['Precision']*msd_pr['Recall'])/(msd_pr['Precision']+msd_pr['Recall']))
    
    # Make ROC dataframe
    msd_roc = pd.DataFrame(list(zip(fpr, tpr)), columns = ['fpr', 'tpr'])
    
    return msd_roc_auc, msd_pr, msd_roc
    
# HYPERPARAMETER TUNING FOR THE MODEL
def tuning(model_not, model, sampled_x, sampled_y, scoring_method):
    # Tune our chosen model using a predefined space of hyperparameters dependent on the model type
    # Tuning will consider our sampling/balancing method as applied to our dataset
    
    print(model)
    print(model_not)
    
    search = None
    
    # Big if/else for our model
    # Nothing for dummy classifier
    
    if model_not == 1:
        # Logistic Regression
        parameters = {'penalty':('l2', None), 'solver':('lbfgs', 'sag', 'liblinear'), 'max_iter':[10, 50, 100, 1000]}
        search = 'random'
        
    elif model_not == 2:
        # Linear Discriminant Analysis
        parameters = {'solver':('svd', 'lsqr', 'eigen')}
        search = 'grid'
        
#    elif model_not == 3:
        # Complement Naive Bayes
#        parameters = {'alpha':[1e-5, 1e-3, 0.1, 0.5, 1], 'norm':(True, False)}
#        search = 'grid'
        
    elif model_not == 3:
        # Decision Tree Classifier
        parameters = {'criterion':('gini', 'entropy', 'log_loss'), 'splitter':('best', 'random'), 'min_samples_split':[2, 5, 10, 20, 40], 'min_samples_leaf':[1, 2, 4, 8, 16, 20]}
        search = 'random'
        
    elif model_not == 4:
        # K-Nearest Neighbours Classifier (k-NN)
        parameters = {'weights':('uniform', 'distance', None), 'leaf_size':list(range(1,50)), 'n_neighbors':list(range(1,30)), 'p':[1, 2]}
        search = 'random'
        
    elif model_not == 5:
        # Support Vector Classifier (SVC)
        parameters = {'C':[0.01, 0.1, 1, 10], 'gamma':('scale', 'auto'), 'kernel':('linear', 'rbf', 'poly')}
        search = 'random'
        
    elif model_not == 6:
        # Bagging Classifier
        parameters = {'n_estimators':[1, 2, 4, 8, 16]}
        search = 'grid'
        
    elif model_not == 7:
        # Random Forest Classifier
        parameters = {'n_estimators':[100, 200, 500, 1000, 2000], 'max_depth':[10, 20, 50, 100], 'min_samples_split':[2, 5, 10], 'min_samples_leaf':[1, 2, 4], 'max_features':('sqrt', 'log2'), 'bootstrap':(True, False)}
        search = 'random'

    elif model_not == 8:
        # Extra Trees Classifier
        parameters = {'n_estimators':[100, 200, 500, 1000, 2000], 'min_samples_leaf':[5, 10, 20], 'max_features':[2, 3, 4]}
        search = 'random'
        
    elif model_not == 9:
        # Gradient Boosting Classifier
        parameters = {'max_depth':[3, 5, 7, 9, 10], 'n_estimators':[1, 2, 5, 10, 20, 50, 100, 200, 500], 'learning_rate':loguniform.rvs(0.01, 1, size=10).tolist()}
        search = 'random'
    
    if search == 'random':
        # Set up randomsearchCV
        clf = RandomizedSearchCV(model, parameters, scoring=scoring_method)
        # run the search
        result = clf.fit(sampled_x, sampled_y)
        print('Best hyperparameters', result.best_params_)
        best_params = result.best_params_
    elif search == 'grid':
        # Set up gridsearchCV
        clf = GridSearchCV(model, parameters, scoring=scoring_method)
        # run the search
        result = clf.fit(sampled_x, sampled_y)
        print('Best hyperparameters', result.best_params_)
        best_params = result.best_params_
    else:
        # We either have the dummy model, or the averaged metric for the meta model
        # No hyperparameter search required
        print('Unable to perform hyperparameter search due to model', model)
        best_params = "none"
    
    # return dict of best hyperparameters for our model
    return best_params
    
# HYPERPARAMETER TUNING FOR THE DATA BALANCING
def bal_tune(model, sampler, samp_not, train_x, train_y, scoring_method):
    # discrep is the ratio of balanced:unbalanced data
    if samp_not in [1, 7, 8, 12, 13]:
        # Random oversampler or Random undersampler, Tomek Links, SMOTE ENN, or SMOTE Tomek
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1]}
        search = 'grid'
    elif samp_not == 2:
        # SMOTE
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1], 'samp__k_neighbors':[1, 3, 5, 10, 20, 40]}
        search = 'random'
    elif samp_not == 3:
        # Borderline SMOTE
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1], 'samp__k_neighbors':[1, 3, 5, 10, 20, 40], 'samp__m_neighbors':[1, 3, 5, 10, 20, 40], 'samp__kind':['borederline-1', 'borderline-2']}
        search = 'random'
    elif samp_not == 4:
        # SVM SMOTE
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1], 'samp__k_neighbors':[1, 3, 5, 10, 20, 40], 'samp__m_neighbors':[1, 3, 5, 10, 20, 40], 'samp__out_step':[0.1, 0.25, 0.5, 0.75, 0.9]}
        search = 'random'
    elif samp_not == 5:
        # K Means SMOTE
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1], 'samp__k_neighbors':[1, 3, 5, 10, 20, 40], 'samp__k_means_estimator':[None, 1, 2, 5, 10, 20], 'samp__density_exponent':['auto', 0.1, 0.25, 0.5, 0.75]}
        search = 'random'
    elif samp_not == 6:
        # ADASYN
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1], 'samp__n_neighbors':[1, 2, 5, 10, 20]}
        search = 'random'
    elif samp_not == 9:
        # Edited Nearest Neighbours
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1], 'samp__n_neighbors':[1, 2, 5, 10, 20], 'samp__kind_sel':['all','mode']}
        search = 'random'
    elif samp_not == 10:
        # Neighbourhood Cleaning Rule
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1], 'samp__n_neighbors':[1, 2, 5, 10, 20], 'samp__threshold_cleaning':[0.1, 0.25, 0.5, 0.75, 0.9]}
        search = 'random'
    elif samp_not == 11:
        # One Sided Selection
        parameters = {'samp__sampling_strategy':['auto',  0.1, 0.25, 0.5, 0.75, 0.9, 1], 'samp__n_neighbors':[1, 2, 5, 10, 20], 'samp__n_seeds_S':[1, 2, 3, 5, 10]}
        search = 'random'
    else:
        # None
        search = 'none'
    
    
    ## GRID SEARCH OR RANDOM SEARCH
    if search == 'random':
        pipe = Pipeline([('samp', sampler), ('model', model)])
        # Set up randomsearchCV
        blf = RandomizedSearchCV(pipe, parameters, random_state=8, scoring=scoring_method)
        # run the search
        result = blf.fit(train_x, train_y)
        print('Best hyperparameters for sampling', result.best_params_)
        params = result.best_params_
    elif search == 'grid':
        pipe = Pipeline([('samp', sampler), ('model', model)])
        # Set up gridsearchCV
        blf = GridSearchCV(pipe, parameters, scoring=scoring_method)
        # run the search
        result = blf.fit(train_x, train_y)
        print('Best hyperparameters for sampling', result.best_params_)
        params = result.best_params_
    else:
        # We either have the dummy model, or the averaged metric for the meta model
        # No hyperparameter search required
        print('Unable to perform hyperparameter search for sampling method as none is being applied.')
        params = None
    
    return params



# PLOTTING
def Plot (pr_metrics, pr_saveas, roc_metrics, roc_auc, roc_saveas, metric_saveas):
    pr_metrics = pr_metrics.drop(columns='Sensitivity')
    # Plot scores from metrics method as pr and roc curves
    sns.set_context('paper')
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(x='Threshold', y='value', hue='variable', data=pd.melt(pr_metrics, 'Threshold'), ci=None)
    ax.set(ylabel='Performance')
    ax.grid()
    ax.legend(title='', bbox_to_anchor=(.5, 1), loc='lower center', ncol=3)
    plt.ylim(0,1)
    plt.xlim(0,1)
    metric_save_pdf = os.path.abspath(metric_saveas+'.pdf')
    metric_save_png = os.path.abspath(metric_saveas+'.png')
    #plt.savefig(metric_save_png, dpi=72, bbox_inches='tight')
    plt.savefig(metric_save_pdf, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.plot([0,1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(roc_metrics['fpr'],roc_metrics['tpr'],label="Model, ROC AUC="+str(round(roc_auc, 3)))
    plt.legend(title='', loc='lower right', ncol=1)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    roc_save_pdf = os.path.abspath(roc_saveas+'.pdf')
    roc_save_png = os.path.abspath(roc_saveas+'.png')
    #plt.savefig(roc_save_png, dpi=72, bbox_inches='tight')
    plt.savefig(roc_save_pdf, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.plot([0,1],[0,0], linestyle='--', label='No Skill')
    pr_auc_score = auc(pr_metrics['Recall'],pr_metrics['Precision'])
    plt.plot(pr_metrics['Recall'],pr_metrics['Precision'], label='Model (PR AUC = '+str(round(pr_auc_score, 3))+')')
    plt.legend(title='', loc='upper right', ncol=1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0,1)
    plt.xlim(0,1)
    pr_save_pdf = os.path.abspath(pr_saveas+'.pdf')
    pr_save_png = os.path.abspath(pr_saveas+'.png')
    #plt.savefig(pr_save_png, dpi=72, bbox_inches='tight')
    plt.savefig(pr_save_pdf, dpi=300, bbox_inches='tight')
    plt.clf()

# FEATURE IMPORTANCE
def feature_importance_determination_and_graphing (model, sampling, train_x, train_y, test_x, test_y, saveas, scoring_method):
    print('Running permutation importance...')   # db
    
    # First apply sampling strategy to our data
    if sampling != 'none':
        train_x, train_y = sampling.fit_resample(train_x, train_y)
        #test_x, test_y = sampling.fit_resample(test_x, test_y)
    
    # Next fit our model to the training data
    model = model.fit(train_x, train_y)
    
    # Now run the permutation feature importance with our testing data
    r = permutation_importance(model, test_x, test_y, random_state=0, scoring=scoring_method)
    print('Finished permutation importance... Now graphing...')   # db
    
    # Format for graphing
    feature_imp = pd.DataFrame(zip(r['importances_mean'], r['importances_std']), columns=['Mean Feature Importance', 'Standard Deviation'])
    feature_imp.index = test_x.columns
    print(feature_imp)   # db
    feature_imp['Abs Importance'] = feature_imp['Mean Feature Importance']
    feature_imp['Abs Importance'] = feature_imp['Abs Importance'].abs()
    feature_imp = feature_imp.sort_values(by='Abs Importance', ascending=False)
    top_30_features = feature_imp[:30]
    top_20_features = feature_imp[:20]
    
    print(top_20_features)   # db
    
    # Graph top 30
    plt.clf()
    plt.figure(figsize=(12, 10))
    plt.barh(top_30_features.index, top_30_features['Mean Feature Importance'], xerr = top_30_features['Standard Deviation'], capsize=2)
    plt.xlabel('Feature Weight')
    plt.ylabel('Base Model Features')
    plt.savefig(saveas + '_top_30_feature_importance.pdf', dpi=300, bbox_inches='tight')
    
    # Graph top 20
    plt.clf()
    plt.figure(figsize=(12, 10))
    plt.barh(top_20_features.index, top_20_features['Mean Feature Importance'], xerr = top_20_features['Standard Deviation'], capsize=2)
    plt.xlabel('Feature Weight')
    plt.ylabel('Base Model Features')
    plt.savefig(saveas + '_top_20_feature_importance.pdf', dpi=300, bbox_inches='tight')
    plt.clf()

# BASE MODEL FITTING
def model_fitting(X, y, auto_fit, i_u_mod, i_u_samp, saveas):
    # auto_fit = T/F to initiate automatic feature fitting
    # feat_x, feat_y = full dataset
    # i_u_mod, i_u_samp = user input selection of model and sampling method
    
    ratio = Counter(y.iloc[:,0])[1]/(len(y))
    print(ratio)
    if ratio < 0.2 or ratio > 0.8:
        scoring_method = 'f1'
        print('Data is imbalanced (', round(ratio*100, 2), '% pos) adjusting model metric to f-score to best assess fit...')
    else:
        scoring_method = 'roc_auc'
        print('Data is balanced', round(ratio*100, 2), '% pos) adjusting model metric to roc-auc to best assess fit...')
    
    if 'SECONDARY_ML_SCORE' in X.columns:
        X = X.drop(columns='SECONDARY_ML_SCORE')
        
    # SPLIT FEATURES INTO TRAIN + TEST
    train_x, test_x, train_y, test_y = train_test_split(X, y, stratify=y, test_size=0.1)
    
    # Fit the models and balancing methods - return the best ones, along with their F-score
    model_max, bal_max, f1_score, model_max_index, bal_max_index = model_fitting_process(train_x, train_y, auto_fit, True, i_u_mod, i_u_samp, scoring_method)
    
    print('Finished model fitting, proceeding with', model_max, 'and', bal_max, 'with an F-metric of', f1_score)
    
    # Hyperparameter tuning - ensure it's not the dummy classifier first
    if model_max_index != 0:
        params = tuning(model_max_index, model_max, train_x, train_y.iloc[:,0], scoring_method)
        
        # Apply hyperparameters to the model
        model_max = model_max.set_params(**params)
    else:
        params = 'none'
        
    
    # Hyperparameter tuning for data balancing method - DO WE WANT TO RUN THIS? MIGHT OVERFIT
    # only run if data is quite imbalanced (i.e. under 20% of underrepresented case)
    #    if bal_max_index != 0:
    #        bal_params = bal_tune(clone(model_max), bal_max, bal_max_index, train_x, train_y)
    #
    #        new_b_params = {}
    #        for k, v in bal_params.items():
    #            new_key = k[6:]
    #            new_b_params[new_key] = v
            
            # Apply hyperparameters to the model
    #        bal_max = bal_max.set_params(**new_b_params)
    #        print('data balancing tuned with', bal_max)
    
    
    unfit_base_model = clone(model_max)
    
    # Now generate metrics for plots
    metrics, roc_metrics, roc_auc = model_metric_generation(train_x, train_y, test_x, test_y,
                            unfit_base_model, bal_max)
    
    # Print some updates to the user
    print('Finished generating metrics. Now plotting...')
    
    ms = ''.join(filter(str.isalnum, str(model_max)))
    ss = ''.join(filter(str.isalnum, str(bal_max)))
    
    save_annotation = str(saveas) + ms + '_' + ss
    Plot(metrics, (save_annotation+'_pr_curve'), roc_metrics, roc_auc, (save_annotation+'_roc_curve'), (save_annotation+'_metrics'))
    
    metrics['F-score'] = 2*((metrics['Precision']*metrics['Recall'])/(metrics['Precision']+metrics['Recall']))
    
    # Feature importance - fit model to training data
    feature_importance_determination_and_graphing(model_max, bal_max, train_x, train_y, test_x, test_y, save_annotation, scoring_method)
    print('Feature importance determination complete')
    
    # Update metric scores
    print("Sensitivity at 0.5: " + str(round(metrics.iloc[50]['Sensitivity'], 2)))
    print("Specificity at 0.5: " + str(round(metrics.iloc[50]['Specificity'], 2)))
    print("Precision at 0.5: " + str(round(metrics.iloc[50]['Precision'], 2)))
    print("Recall at 0.5: " + str(round(metrics.iloc[50]['Recall'], 2)))
    print("Maximised F-score of", metrics.loc[metrics['F-score'].idxmax()]['F-score'].round(2), "at a threshold of", metrics.loc[metrics['F-score'].idxmax()]['Threshold'].round(2), "Recall:", metrics.loc[metrics['F-score'].idxmax()]['Recall'].round(2), "Specificity:", metrics.loc[metrics['F-score'].idxmax()]['Specificity'].round(2), "Precision:", metrics.loc[metrics['F-score'].idxmax()]['Precision'].round(2), "Sensitivity:", metrics.loc[metrics['F-score'].idxmax()]['Sensitivity'].round(2))
    
    return model_max, unfit_base_model, bal_max, metrics, train_x, train_y, params, scoring_method
    
# META MODEL FITTING
def meta_model_fitting(base_model, base_bal, feat_x, feat_y, user_feat, i_u_mod, i_u_samp, saveas, scoring_method):
    
    # STEP 1 - SPLIT UP THE DATA APPROPRIATELY
    print('Initial length of features - x:', len(feat_x), ', y:', len(feat_y))   # db
    
    # Check if there's a user-defined holdout test set for testing of the ensemble model
    if 'SOURCE' in user_feat.columns:
        # Pull it
        holdout = user_feat[user_feat['SOURCE'] != 'PhosphositePlus']
        leftover = user_feat[usser_feat['SOURCE'] == 'PhosphositePlus']
        
        # Create a new df of user-defined test set containing holdout values
        meta_test_x = pd.DataFrame(holdout[['SECONDARY_ML_SCORE']])
        meta_test_x_for_base_mod = pd.DataFrame(feat_x.lpc[holdout.index])
        meta_test_y = pd.DataFrame(feat_y.loc[holdout.index])
        
        # Now define training data for the base and meta-learning model from the points remaining
        temp_train_x = pd.DataFrame(feat_x.loc[leftover.index])
        temp_train_y = pd.DataFrame(feat_y.loc[leftover.index])
        
    else:
        # User hasn't included a 'SOURCE' column, meaning there's no user-defined holdout set
        # Create a test set from our full feature dataset
        temp_x, meta_test_x_for_base_mod, temp_y, meta_test_y = train_test_split(feat_x, feat_y, stratify = feat_y, test_size = 0.1)
        meta_test_x = pd.DataFrame(meta_test_x_for_base_mod[['SECONDARY_ML_SCORE']])
        
    print('Test set contains x:', len(meta_test_x_for_base_mod), ', y:', len(meta_test_y))
    print('After removing the test set, remaining features are - x:', len(temp_x), ', y:', len(temp_y))
    
    # Now split the temp set into training sets for the base model and the ensemble model
    base_train_x, meta_train_x, base_train_y, meta_train_y = train_test_split(temp_x, temp_y, test_size = 0.5, stratify = temp_y)
    
    print('Base train x set:', len(base_train_x))
    print('Meta train x set:', len(meta_train_x))
    
    # STEP 2 - GENERATE ML SCORES FOR THE BASE ML MODEL FOR ENSEMBLE TRAINING + TESTING DATA
    # Drop the SECONDARY_ML_SCORE from the dataset for the base model predictor
    base_train_x = base_train_x.drop(columns=['SECONDARY_ML_SCORE'])
    
    meta_combo_train_x = pd.DataFrame(meta_train_x[['SECONDARY_ML_SCORE']])
    meta_train_x = meta_train_x.drop(columns=['SECONDARY_ML_SCORE'])   # Can drop secondary score from meta now that it's safely in a new df
    
    if base_bal != "none":
        print('Balancing the dataset via', base_bal)
        base_train_x, base_train_y = base_bal.fit_resample(base_train_x, base_train_y)
    else:
        print('Did not balance the dataset as base_bal =', base_bal)
    
    print('Base model:', base_model)
    base_model = base_model.fit(base_train_x, base_train_y)
    print('Base train x set:', len(base_train_x))
    
    # Now predict the base model scores for the meta training data
    base_train_scores = base_model.predict_proba(meta_train_x)
    base_train_scores = base_train_scores[:, [1]]
    meta_combo_train_x['PRIMARY_ML_SCORE'] = base_train_scores
    
    # Finally predict the base model scores for the meta test set
    meta_test_x_for_base_mod = meta_test_x_for_base_mod.drop(columns=['SECONDARY_ML_SCORE'])
    base_test_scores = base_model.predict_proba(meta_test_x_for_base_mod)
    base_test_scores = base_test_scores[:, [1]]
    meta_test_x['PRIMARY_ML_SCORE'] = base_test_scores
    
    
    # STEP 3 - FIT DIFFERENT ENSEMBLE LEARNING MODELS, RETURN BEST ONE
    meta_model_max, meta_bal_max, meta_f1_score, meta_model_max_index, meta_bal_max_index = model_fitting_process(meta_combo_train_x, meta_train_y, True, False, None, None, scoring_method)
    
    print('Finished model fitting, proceeding with', meta_model_max, 'and', meta_bal_max, 'with an', scoring_method, 'of', meta_f1_score)
    
    # STEP 4 - HYPERPARAMETER TUNING OF THE ENSEMBLE MODEL
    
    # Hyperparameter tuning - ensure it's not the dummy classifier first
    if meta_model_max_index != 0 and type(meta_model_max) != str:
        meta_params = tuning(meta_model_max_index, meta_model_max, meta_combo_train_x, meta_train_y, scoring_method)
        
        # Apply hyperparameters to the model
        meta_model_max = meta_model_max.set_params(**meta_params)
    
    
        unfit_meta_model = clone(meta_model_max)
    
    else:
        unfit_meta_model = meta_model_max
        meta_params = 'none'
    
    
    # STEP 5 - GENERATE METRICS FOR THE BEST ENSEMBLE LEARNING METHOD
    # Now generate metrics for plots
    meta_metrics, meta_roc_metrics, meta_roc_auc = model_metric_generation(meta_combo_train_x,
                                                                           meta_train_y,
                                                                           meta_test_x,
                                                                           meta_test_y,
                                                                           unfit_meta_model,
                                                                           meta_bal_max)
    
    # Print some updates to the user
    print('Finished generating metrics. Now plotting...')
    
    ms = ''.join(filter(str.isalnum, str(meta_model_max)))
    ss = ''.join(filter(str.isalnum, str(meta_bal_max)))
    
    save_annotation = str(saveas) + '_META_' + ms + '_' + ss
    Plot(meta_metrics, (save_annotation+'_pr_curve'), meta_roc_metrics, meta_roc_auc, (save_annotation+'_roc_curve'), (save_annotation+'_metrics'))
    
    meta_metrics['F-score'] = 2*((meta_metrics['Precision']*meta_metrics['Recall'])/(meta_metrics['Precision']+meta_metrics['Recall']))
    
    # Update metric scores
    print("Sensitivity at 0.5: " + str(round(meta_metrics.iloc[50]['Sensitivity'], 2)))
    print("Specificity at 0.5: " + str(round(meta_metrics.iloc[50]['Specificity'], 2)))
    print("Precision at 0.5: " + str(round(meta_metrics.iloc[50]['Precision'], 2)))
    print("Recall at 0.5: " + str(round(meta_metrics.iloc[50]['Recall'], 2)))
    print("Maximised F-score of", meta_metrics.loc[meta_metrics['F-score'].idxmax()]['F-score'].round(2), "at a threshold of", meta_metrics.loc[meta_metrics['F-score'].idxmax()]['Threshold'].round(2), "Recall:", meta_metrics.loc[meta_metrics['F-score'].idxmax()]['Recall'].round(2), "Specificity:", meta_metrics.loc[meta_metrics['F-score'].idxmax()]['Specificity'].round(2), "Precision:", meta_metrics.loc[meta_metrics['F-score'].idxmax()]['Precision'].round(2), "Sensitivity:", meta_metrics.loc[meta_metrics['F-score'].idxmax()]['Sensitivity'].round(2))
    
    return meta_model_max, unfit_meta_model, meta_bal_max, meta_metrics, base_train_x, base_train_y, meta_combo_train_x, meta_train_y, meta_test_x, meta_test_y

    
    
