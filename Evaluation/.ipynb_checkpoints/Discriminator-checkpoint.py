import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import datasets
from sklearn.model_selection import train_test_split

import Evaluation.fit_predict

def discriminator(path,real_founders,fake_founders,real_labels,fake_labels,hparams):
    
    ## Define scale boolean. If True, input SNPs will be scaled. If False, input SNPs won't be scaled
    scale = False

    ## Specify number of jobs to run in parallel when tuning the hyperparameters
    # If set to 1, only one processor will be used
    # If set to -1, all processors will be used
    NUM_JOBS_PARALLEL = -1

    ## Specify the scoring metric to use for evaluating best hyperparameters set
    # Use any scoring methods supported by sklearn: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    SCORING_FOR_SEARCH = 'balanced_accuracy'

    ## Output file path
    output_path = path

    ## Dict with classification method names as keys and dict with hyerparameter settings to try as values
    CLASSIFIERS = {
        "Logistic Regression": {
            "method": LogisticRegression(random_state=0), 
            "param_grid": {
                "penalty": ['l2', 'none'], 
                "C": [0.001,0.1], 
                "class_weight": [None, 'balanced'], 
                "solver": ['newton-cg', 'lbfgs'], 
                "max_iter": [100], 
               }
        }, 
        "K-Nearest Neighbors Classifier": {
            "method": KNeighborsClassifier(), 
            "param_grid": {
                "n_neighbors": [2, 4, 10], 
                "weights": ['uniform', 'distance'], 
                "leaf_size": [10, 20, 30],
                "p": [1,2]
                }
        },

        "Multi-layer Perceptron (Non-linear)": {
            "method": MLPClassifier(random_state=0),
            "param_grid": {
            }
        }
        
    }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Concat real and fake samples and labels
    founders = torch.cat([real_founders, fake_founders], 0)
    ancestries = torch.cat([real_labels, fake_labels], 0)
    # Add labels to the dataset. If the sample is generated 1, if not 0
    real = torch.zeros(len(real_labels))
    fake = torch.ones(len(fake_labels))
    Generated = torch.cat([real, fake], 0)
    labels = torch.cat([ancestries.unsqueeze(1), Generated.unsqueeze(1)], 1)
    labels = pd.DataFrame(labels.numpy())
    labels.columns = ['Ancestry','Generated']
 

    ## For each regressor to fit
    for classifier_name in CLASSIFIERS:
        gridsearch_object = GridSearchCV(
            estimator = CLASSIFIERS[classifier_name]["method"],
            param_grid = CLASSIFIERS[classifier_name]["param_grid"],
            n_jobs = NUM_JOBS_PARALLEL,
            scoring = SCORING_FOR_SEARCH,
            refit = True,
            cv = 5,
            verbose = 0
        )
        
        ## Create empty dataframe that will store performance metrics for train and test sets of each seed
        # Columns: name of the classifier, best parameters, score and mean square error for train and test sets
        columns_perf = ["Classifier name", "Best estimator", 
                    "Accuracy (train)", "Precision (train)", "Recall (train)", "F1 (train)", "Balanced accuracy (train)",
                    "Accuracy (test)", "Precision (test)", "Recall (test)", "F1 (test)", "Balanced accuracy (test)",
                    "Time (train)"]
        Performance = pd.DataFrame(columns=columns_perf)   
        
        for i in range(5):
            ## Separate data into train, validation and test sets
            # Different seed each time
            X_train, X_test, y_train, y_test = train_test_split(founders, labels, test_size=0.33) 
            
            ## If scale is set to true, scale SNPs data
            if scale == True:
                scaler = StandardScaler()
                scaler = scaler.fit(X_train)

                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            
            print('-> Fitting and evaluating: {}'.format(classifier_name))

            ## Perform hyperparameter tuning and refit over best hyperparameter configuration to obtain:
            # - Predictions of the classifier on the test set
            # - Performance of the classifier on train and test sets 
            predictions, performance = Evaluation.fit_predict.predict_evaluate(classifier_name, gridsearch_object, 
                                                        X_train=X_train, y_train=y_train['Generated'].to_numpy(), 
                                                        X_test=X_test, y_test=y_test['Generated'].to_numpy(), 
                                                        ancestry=y_test['Ancestry'].to_numpy())

            ## Save performances as a new row of the dataframe with the performances
            Performance = Performance.append(performance)
        
        # Compute mean and sd of the accuracies of differents seeds and store the results in a dataframe
        columns_perf = ["Classifier name","Mean Accuracy (train)","Sd Accuracy (train)","Mean Accuracy (test)","Sd Accuracy (test)","time"]
        ## Append performance results
        performance_mean_var = pd.DataFrame([[classifier_name,Performance["Accuracy (train)"].mean(), Performance["Accuracy (train)"].std(),Performance["Accuracy (test)"].mean(), Performance["Accuracy (test)"].std(),Performance["Time (train)"].mean()]], columns=columns_perf)
        # Save results
        if hparams['Save_discriminator']:
            Evaluation.fit_predict.save_results(output_path, predictions=predictions, performance=performance_mean_var)
            print('Output saved')