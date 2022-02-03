import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

def evaluate(true_labels, predictions):
    '''
    Objective:
        - Calculate performance metrics given true labels and predictions.
    Inputs:
        - true_labels: vector with true labels.
        - predictions: vector with predictions.
    '''
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)

    return accuracy, precision, recall, f1, balanced_accuracy


def predict_evaluate(classifier_name, gridsearch_object, X_train, y_train, X_test, y_test, ancestry):
    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Objective:
        - Perform hyperparameter tuning for a given classifier. Refit over best hyperparameter configuration and evalutate the model.
    Inputs:
        - classifier_name: name of the classifier to train/test (string).
        - gridsearch_object: gridsearch object to be trained.
        - X_train: SNP sequences in train set.
        - y_train: generated label in train set.
        - X_test: SNP sequences in test set.
        - y_test: generated label in test set.
        - ancestry: ancestry in test set.
    Outputs:
        - Dataframe containing a row with predictions for test set.
        - Dataframe containing a row with performance metrics for train and test sets.
    '''
    
    ## Create empty dataframe that will store the predictions for test set
    # Columns: name of the classifier, best parameters and predictions
    columns_pred = ["Classifier name", "Best estimator", "Imputed true (test)", "Imputed predictions (test)", "Ancestry"]
    predictions = pd.DataFrame(columns=columns_pred)
    
    ## Create empty dataframe that will store performance metrics for train and test sets
    # Columns: name of the classifier, best parameters, score and mean square error for train and test sets
    columns_perf = ["Classifier name", "Best estimator", 
                    "Accuracy (train)", "Precision (train)", "Recall (train)", "F1 (train)", "Balanced accuracy (train)",
                    "Accuracy (test)", "Precision (test)", "Recall (test)", "F1 (test)", "Balanced accuracy (test)",
                    "Time (train)"]
    
    performance = pd.DataFrame(columns=columns_perf)
    
    ## Fit the regessors and measure fitting time
    import time
    start = time.time()
    gridsearch_object = gridsearch_object.fit(X_train, y_train)
    end = time.time()
    time = end - start
    
    ## Predict the y label for train and test sets
    y_train_pred = gridsearch_object.predict(X_train)
    y_test_pred = gridsearch_object.predict(X_test)
    
    ## Calculate R2 and MSE performance metrics for train and test sets
    accuracy_train, precision_train, recall_train, f1_train, balanced_accuracy_train = evaluate(y_train, y_train_pred)
    accuracy_test, precision_test, recall_test, f1_test, balanced_accuracy_test = evaluate(y_test, y_test_pred)
       
    ## Obtain best configuration estimator
    best_est = gridsearch_object.best_estimator_
        
    ## Append prediction results
    predictions = pd.DataFrame([[classifier_name, best_est, y_test, y_test_pred, ancestry]], columns=columns_pred)

    ## Append performance results
    performance = pd.DataFrame([[classifier_name, best_est, 
                                 accuracy_train, precision_train, recall_train, f1_train, balanced_accuracy_train, 
                                 accuracy_test, precision_test, recall_test, f1_test, balanced_accuracy_test,
                                 time]], columns=columns_perf)

    return predictions, performance


def save_results(output_path, predictions, performance):
    '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Objectives:
        - Save new row with predictions of a particular classifier in the given path.
        - Save new row with performance of a particular classifier in the given path.
    Inputs:
        - output_path: file path to the dataframes.
        - predictions: dataframe with a single row containing predictions for a specific method.
        - performance: dataframe with a single row containing performance results for a specific method.
    '''

    ## If it does not already exist, create directory that will contain output results
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    ## Define path of the predictions and performance dataframes to be saved
    performance_name = output_path + '/' + 'performance.csv'
    predictions_name = output_path +'/' + 'predictions.csv'
    
    if not os.path.exists(performance_name):
        performance.to_csv(performance_name, index=False)
    else:
        ## Read dataframe, add new row to dataframe and save results
        performance_df = pd.read_csv(performance_name)
        performance_df = performance_df.append(performance)
        performance_df.to_csv(performance_name, index=False)
        
    ## If the prediction dataframe does not exist, save dataframe with predictions
    if not os.path.exists(predictions_name):
        predictions.to_csv(predictions_name, index=False)
    else:
        ## Read dataframe, add new row and save results
        predictions_df = pd.read_csv(predictions_name)
        predictions_df = predictions_df.append(predictions)
        predictions_df.to_csv(predictions_name, index=False)