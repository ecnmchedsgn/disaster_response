# import packages for data loading 
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# import packages for text processing
import nltk
nltk.download(['punkt', 'wordnet'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# import packages for building ML model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# import packages for training and evaluating ML model
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, precision_recall_fscore_support, multilabel_confusion_matrix, confusion_matrix

# import packages to perform optimization
from sklearn.model_selection import  KFold


# import packages for upsample
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import time
import statistics
import math
import matplotlib.pyplot as plt

# import packages to save model
import pickle

def load_data(database_filepath, optimize):
    '''
    Input:
        database_filepath   filepath to SQL database
        optimize            indicator whether loading dataset for optimization or training models
                            optimize = 1 means for optimization and dataset is sub-sampled for faster optimization

    Output:
        X               (pandas Series) messages to categorize (str in English)
        Y               (pandas dataframe) 36 category columns for the messages
        category_names  (list) 36 category names

    Description:
        Load data from SQL database, extract messages, category names and how each message is categorized
    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('categorized_messages', engine) 

    # include for fast optimization or training, subsample dataset
    if optimize == 1:
        df = df.sample(frac=0.1, replace=False, random_state=1)
    else:
        pass
        #df = df.sample(frac=0.2, replace=False, random_state=1)
    
    X = df['message']
    Y = df.iloc[:,4:]

    print('Training predictor for {} labels'.format(Y.shape[1])) 

    category_names = list(Y.columns.values)

    return X, Y, category_names



def tokenize(text):
    '''
    Function that tokenizes character sequence (chops down, remove punctuation, lower case, etc.)
    INPUT: 
    text            (Str) input text 

    OUTPUT:
    clean_tokens    (list) cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_optimize_model(X_train, y_train, X_test, y_test):
    '''
    Description:
        Cross-validated search for best hyperparameters for pipeline (1 pipeline for each label)

    INPUT:
    X_train         (pandas dataframe) training dataset to optimize hyperparameters
    y_train         (pandas dataframe) training target label to optimize hyperparameters
    X_test          (pandas dataframe) testing dataset to optimize hyperparameters
    y_test          (pandas dataframe) testing target label to optimize hyperparameters

    OUTPUT:
    pipeline_list               (list) of pipelines (1 for each label) with optimized hyper-parameters
    threshold_pipeline_list     (list) of thresholds (1 for each label) for prediction; when predicted probability from pipeline is above threshold, label is positive

    '''

    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)

    # build pipeline 
    pipeline_list, threshold_pipeline_list,threshold_std_pipeline_list = [], [],[]
    verbose = 2
    n_splits = 5
    for ind in range(y_train.shape[1]):

        # for each lable, create a list of pipelines for optimization
        # the list includes different pipelines for optimization
        pipeline_optimize_label = []
        
        # create list of parameters for tracking parameters
        n_estimators_optimize, min_samples_split_optimize,min_samples_leaf_optimize, sample_ratio_optimize  = [], [], [],[]

        # find ratio of positive to negative labels
        ratio_pos_to_neg = y_train.iloc[:,ind].value_counts()[1]/y_train.iloc[:,ind].value_counts()[0]
        sample_ratio_range = [0.1,0.2,0.5,1]
        # if pos_label is minority, choose sample_ratio larger than minority ratio (aka OVER-sampling minority class)
        if ratio_pos_to_neg < 1:
            sample_ratio_range = [x for x in sample_ratio_range if x>ratio_pos_to_neg]
        else: #if pos_label is majority, then sample_ratio = 1
            sample_ratio_range = [1]

        # build a list of pipelines for the current label
        # using hyperparameters we'd like to optimize
        seed = 100
        for n_estimators in [100,200,500,1000]:
            for min_samples_split in [2,5,10]:
                for min_samples_leaf in [1,2,5]:
                    for sample_ratio in sample_ratio_range:
                        pipeline_optimize_label.append(make_pipeline_imb(
                                CountVectorizer(tokenizer=tokenize),
                                TfidfTransformer(),
                                SMOTE(k_neighbors=1, random_state=seed,sampling_strategy=sample_ratio),
                                RandomForestClassifier(n_jobs = 8, n_estimators = n_estimators, min_samples_split = min_samples_split, 
                                                            min_samples_leaf = min_samples_leaf)))

                        # lists to record corresponding hyperparameters for optimization
                        n_estimators_optimize.append(n_estimators)
                        min_samples_split_optimize.append(min_samples_split)
                        min_samples_leaf_optimize.append(min_samples_leaf)
                        sample_ratio_optimize.append(sample_ratio)


        # record best F score from optimization
        fscore_mean_optimize,fscore_std_optimize = [],[]
        fscore_mean_threshold_optimize,fscore_std_threshold_optimize = [],[]
        threshold_mean_optimize, threshold_std_optimize = [], []

        # Optimize pipeline for current label and output parameter space
        col = y_train.columns[ind]
        y_train_col = y_train[col]
        print('Optimizing column {} with {} parameter combinations, each cross-validated with {} folds'.format(col, len(n_estimators_optimize), n_splits))

        # start timer for current label
        start_time  = time.time()

        for ind_optimize in range(len(pipeline_optimize_label)):

            # select pipeline and corresponding sample_ratio for this pipeline
            pipeline_ind = pipeline_optimize_label[ind_optimize]
            sample_ratio_ind = sample_ratio_optimize[ind_optimize]

            # record F scores
            fscore_mean, fscore_mean_threshold = 0, 0
            fscore_list,fscore_list_threshold = [],[]
            threshold_optimal_list = []

            # split train dataset n-folds for training
            kf = KFold(n_splits=n_splits)

            for train_index, test_index in kf.split(X_train):
                
                X_train_cv = X_train.loc[train_index]
                y_train_cv = y_train_col.loc[train_index]

                ratio_pos_to_neg_cv = y_train_cv.value_counts()[1]/y_train_cv.value_counts()[0]
                # if there is only one label in train dataset, no need to train and go on to next fold
                if y_train_cv.nunique() == 1:
                    continue
                
                # if pos-negative ratio is greater than specified ratio, minority-case will be down-sampled
                # go on to next fold
                elif ((ratio_pos_to_neg_cv > sample_ratio_ind) & (ratio_pos_to_neg_cv < 1)): 
                    continue

                else:
                    # train pipeline
                    pipeline_ind.fit(X_train_cv, y_train_cv)

                    # test on original not upsampled data
                    y_pred = pipeline_ind.predict(X_test)

                    # find F-score for positive label = 1
                    classify = precision_recall_fscore_support(y_test.loc[:,col], 
                                                        y_pred, average='binary', pos_label = 1, zero_division = 0)
                    fscore_mean += classify[2]
                    fscore_list.append(classify[2])

                    # Optimize threshold for labeling F-score
                    threshold_list = np.linspace(0.001,1,200)
                    classify_threshold = []
                    predicted_proba = pipeline_ind.predict_proba(X_test)
                    for threshold in threshold_list:

                        # predict label as 1 when probability is above threshold and find F1-score 
                        y_pred = (predicted_proba[:,1] >= threshold).astype('int')

                        classify = precision_recall_fscore_support(y_test.loc[:,col], 
                                                            y_pred, average='binary', pos_label = 1, zero_division = 0)
                        classify_threshold.append(classify[2])
                    # find threshold value that gives best F-score
                    threshold_optimal = threshold_list[classify_threshold.index(max(classify_threshold))]

                    # save max F score and optimal threshold
                    fscore_mean_threshold += max(classify_threshold)
                    fscore_list_threshold.append(max(classify_threshold))
                    threshold_optimal_list.append(threshold_optimal)

            # calculate Cross-calidatae F-scores and optimal thresholds
            fscore_mean /= n_splits
            fscore_std = statistics.stdev(fscore_list)
            fscore_mean_threshold /= n_splits
            fscore_std_threshold = statistics.stdev(fscore_list_threshold)
            threshold_mean = sum(threshold_optimal_list)/len(threshold_optimal_list)
            threshold_std = statistics.stdev(threshold_optimal_list)

            # Output metrics for current parameter choice
            if verbose >= 3:
                print('F-score of {} +- {}'.format(round(fscore_mean,3), round(fscore_std,3)))
                print('F-score (optimize threshold) of {} +- {}'.format(round(fscore_mean_threshold,3), 
                                                                                            round(fscore_std_threshold,3)))
                print('Best thresholds that achieve max F1 for each CV fold: {}'.format(threshold_optimal_list))
                print('Thresholds that achieve max F1: {} +- {}'.format(round(threshold_mean,3), 
                                                                                            round(threshold_std,3)))

            # Record
            fscore_mean_optimize.append(fscore_mean)
            fscore_std_optimize.append(fscore_std)
            fscore_mean_threshold_optimize.append(fscore_mean_threshold)
            fscore_std_threshold_optimize.append(fscore_std_threshold)
            threshold_mean_optimize.append(threshold_mean)
            threshold_std_optimize.append(threshold_std)

            # output elapsed time 3 times 
            if ind_optimize == math.ceil(1/3 * len(pipeline_optimize_label)):
                elapsed_time = time.time() - start_time
                print('Done 1/3 optimization with {} elapsed minutes'.format( round(elapsed_time/60,2)))
            elif ind_optimize == math.ceil(2/3 * len(pipeline_optimize_label)):
                elapsed_time = time.time() - start_time
                print('Done 2/3 optimization with {} elapsed minutes'.format( round(elapsed_time/60,2)))
            elif ind_optimize == len(pipeline_optimize_label) - 1:
                elapsed_time = time.time() - start_time
                print('Finished optimization with {} elapsed minutes'.format( round(elapsed_time/60,2)))

        # after all parameter space searched, find best f-score and corresponding hyperparametes for current label
        print('-'*40)
        best_threshold = 0
        if max(fscore_mean_optimize) > max(fscore_mean_threshold_optimize):
            best_fscore = max(fscore_mean_optimize)
            n_estimators_label = n_estimators_optimize[fscore_mean_optimize.index(best_fscore)]
            min_samples_split_label = min_samples_split_optimize[fscore_mean_optimize.index(best_fscore)]
            min_samples_leaf_label = min_samples_leaf_optimize[fscore_mean_optimize.index(best_fscore)]
            sample_ratio_label = sample_ratio_optimize[fscore_mean_optimize.index(best_fscore)]

            print('Best F1-score for {} is {} with n_estimators = {},  min_samples_split = {}, min_samples_leaf = {}, sample_ratio = {}'.format(col, round(best_fscore,3),
            n_estimators_label, min_samples_split_label, min_samples_leaf_label, sample_ratio_label))

        else:
            best_fscore = max(fscore_mean_threshold_optimize)
            n_estimators_label = n_estimators_optimize[fscore_mean_threshold_optimize.index(best_fscore)]
            min_samples_split_label = min_samples_split_optimize[fscore_mean_threshold_optimize.index(best_fscore)]
            min_samples_leaf_label = min_samples_leaf_optimize[fscore_mean_threshold_optimize.index(best_fscore)]
            sample_ratio_label = sample_ratio_optimize[fscore_mean_threshold_optimize.index(best_fscore)]
            print('Best F1-score (with threshold optimiation) for {} is {} with n_estimators = {},  min_samples_split = {}, min_samples_leaf = {}, sample_ratio = {}'.format(
                col, round(best_fscore,3), n_estimators_label, min_samples_split_label, min_samples_leaf_label, sample_ratio_label))

            threshold_label = threshold_mean_optimize[fscore_mean_threshold_optimize.index(max(fscore_mean_threshold_optimize))]
            threshold_std_label = threshold_std_optimize[fscore_mean_threshold_optimize.index(max(fscore_mean_threshold_optimize))]
            print('It is achieved via threshold of {} +- {}'.format(round(threshold_label,3), round(threshold_std_label,4)))

            best_threshold = 1 #update to indicate best F-score comes with optimizing threshold

        # select best parameters and train the pipeline for current label
        pipeline_label = make_pipeline_imb(
                            CountVectorizer(tokenizer=tokenize),
                            TfidfTransformer(),
                            SMOTE(k_neighbors=1, random_state=seed,sampling_strategy=sample_ratio_label),
                            RandomForestClassifier(n_jobs = 8, n_estimators = n_estimators_label, min_samples_split = min_samples_split_label, 
                                                        min_samples_leaf = min_samples_leaf_label))

        pipeline_label.fit(X_train, y_train_col)

        # append best pipeline for the current label to list of pipelines
        pipeline_list.append(pipeline_label)
        threshold_pipeline_list.append(threshold_label)
        threshold_std_pipeline_list.append(threshold_std_label)

        # output results in csv format
        cv_results = pd.DataFrame()
        cv_results['n_estimators'] = n_estimators_optimize
        cv_results['min_samples_split'] = min_samples_split_optimize
        cv_results['min_samples_leaf'] = min_samples_leaf_optimize
        cv_results['sample_ratio'] = sample_ratio_optimize
        cv_results['fscore_mean'] = fscore_mean_optimize
        cv_results['fscore_std'] = fscore_std_optimize
        cv_results['fscore_threshold_mean'] = fscore_mean_threshold_optimize
        cv_results['fscore_threshold_std'] = fscore_std_threshold_optimize
        cv_results['threshold_mean'] = threshold_mean_optimize
        cv_results['threshold_std'] = threshold_std_optimize

        name_csv = 'cv/CVOptimize_'+col+'_v1.csv'
        cv_results.to_csv(name_csv)

        # output gridsearch results in figure
        if verbose >= 2:
            # select F-score for variable n_estimators, fixed other parameters

            # find indices for specific parameters
            min_split_select = [i for i, x in enumerate(min_samples_split_optimize) if x == min_samples_split_label]
            min_leaf_select = [i for i, x in enumerate(min_samples_leaf_optimize) if x == min_samples_leaf_label]
            sample_ratio_select = [i for i, x in enumerate(sample_ratio_optimize) if x == sample_ratio_label]
            n_estimators_select = [i for i, x in enumerate(n_estimators_optimize) if x == n_estimators_label]

            fig = plt.figure()
            # combine to find indices for varialbe n_estimators
            ind_select = [value for value in min_split_select if value in min_leaf_select]
            ind_select = [value for value in ind_select if value in sample_ratio_select]
            plt.subplot(2,2,1)
            if best_threshold:
                plt.plot([n_estimators_optimize[ind] for ind in ind_select] ,[fscore_mean_threshold_optimize[ind] for ind in ind_select])
            else:
                plt.plot([n_estimators_optimize[ind] for ind in ind_select] ,[fscore_mean_optimize[ind] for ind in ind_select])
            plt.xlabel('number of estimators')
            plt.title(col)

            ind_select = [value for value in n_estimators_select if value in min_leaf_select]
            ind_select = [value for value in ind_select if value in sample_ratio_select]
            plt.subplot(2,2,2)
            if best_threshold:
                plt.plot([min_samples_split_optimize[ind] for ind in ind_select] ,[fscore_mean_threshold_optimize[ind] for ind in ind_select])
            else:
                plt.plot([min_samples_split_optimize[ind] for ind in ind_select] ,[fscore_mean_optimize[ind] for ind in ind_select])
            plt.xlabel('Min samples for split')
            plt.title(col)

            ind_select = [value for value in n_estimators_select if value in min_split_select]
            ind_select = [value for value in ind_select if value in sample_ratio_select]
            plt.subplot(2,2,3)
            if best_threshold:
                plt.plot([min_samples_leaf_optimize[ind] for ind in ind_select] ,[fscore_mean_threshold_optimize[ind] for ind in ind_select])
            else:
                plt.plot([min_samples_leaf_optimize[ind] for ind in ind_select] ,[fscore_mean_optimize[ind] for ind in ind_select])
            plt.xlabel('Min samples for Leaf')
            plt.title(col)

            ind_select = [value for value in n_estimators_select if value in min_split_select]
            ind_select = [value for value in ind_select if value in min_leaf_select]
            plt.subplot(2,2,4)
            if best_threshold:
                plt.plot([sample_ratio_optimize[ind] for ind in ind_select] ,[fscore_mean_threshold_optimize[ind] for ind in ind_select])
            else:
                plt.plot([sample_ratio_optimize[ind] for ind in ind_select] ,[fscore_mean_optimize[ind] for ind in ind_select])
            plt.xlabel('Upsample Ratio')
            plt.title(col)

            plt.tight_layout()
            #plt.show()
            name_fig = 'cv/CVOptimize_'+col+'_v1.png'
            fig.savefig(name_fig)


    return pipeline_list, threshold_pipeline_list

def train_model(X_train, y_train, model):
    '''
    Description:
        Train pipelines (1 for each label)

    INPUT:
    X_train         (pandas dataframe) training dataset to train model
    y_train         (pandas dataframe) training target label to train model
    model           (list) of pipelines (1 for each label) with certain hyperparameters

    OUTPUT:
    model           (list) of pipelines (1 for each label) with certain hyperparameters, and trained using training dataset and target labels

    '''

    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)

    # train pipeline 
    for ind in range(len(model)):
        
        y_train_label = y_train.iloc[:,ind]

        model[ind].fit(X_train, y_train_label)

    return model


def evaluate_model(model, threshold_list, X_test, Y_test, category_names):
    '''
    Evaluate and print-out performance in terms of precision, recall and F1 score
    These metrics are printed out for (1) binary labels (2) positive labels (3) macro-averaged positive label (4) micro-averaged for positive label

    INPUT:
    model               (list) of pipelines (1 for each label) with certain hyperparameters, and trained using training dataset and target labels
    threshold_list      (list) of thresholds (1 for each label) for prediction; when predicted probability from pipeline is above threshold, label is positive
    X_test              (pandas dataframe) testing dataset to test model
    Y_test              (pandas dataframe) testing target labels to test model
    category_names      (list) of label/category  names



    '''
    # make predictions
    y_pred = pd.DataFrame()
    for ind in range(0,len(Y_test.columns)):
        predicted_proba = model[ind].predict_proba(X_test)

        threshold_label = threshold_list[ind]

        y_pred_label = (predicted_proba[:,1] >= threshold_label).astype('int')
        
        # column  name
        col = category_names[ind]

        # add to y_pred
        y_pred.loc[:,col] = y_pred_label

    # print out results for each category
    # for each label, output classification report
    for ind in range(0,len(Y_test.columns)):
        col = Y_test.columns.values[ind]
        print('classification report for category: '+col)
        print(classification_report(Y_test.loc[:,col], y_pred.loc[:,col])) 
    
    # find overall classifier performance (macro average & treating multi-classifier as 3 labels)
    # the multi-label part
    precision = []
    recall = []
    fscore = []
    for ind in range(0,len(Y_test.columns)):
    
        #print(ind)
        #print(y_test_downsampled.columns.values[ind])
        col = Y_test.columns.values[ind]
        classify = precision_recall_fscore_support(Y_test.loc[:,col], 
                                                y_pred.loc[:,col], average='binary', pos_label = 1, zero_division=0)
        precision.append(classify[0])
        recall.append(classify[1])
        fscore.append(classify[2])
    
    # macro average
    precision_macro = sum(precision)/len(precision)
    recall_macro = sum(recall)/len(recall)
    fscore_macro = sum(fscore)/len(fscore)
    print('Macro averaged precion: {}, recall: {}, fscore: {}'.format(precision_macro, recall_macro, fscore_macro))


    # find overall classifier performance (micro average)
    # multi-label part
    confusion_matrix_alllabels = multilabel_confusion_matrix(Y_test, y_pred)
    TN, FN, FP, TP = 0, 0, 0, 0
    for ind in range(0,len(Y_test.columns)):
        TN += confusion_matrix_alllabels[ind][0,0] # true negative
        FN += confusion_matrix_alllabels[ind][1,0] #false negative
        FP += confusion_matrix_alllabels[ind][0,1] #false postitive
        TP += confusion_matrix_alllabels[ind][1,1] #true positive (for positive label = 1)
    
    precision_micro = TP/(TP+FP)
    recall_micro = TP/(TP+FN)
    fscore_micro = 2*precision_micro*recall_micro/(recall_micro+precision_micro)
    print('Micro averaged precion: {}, recall: {}, fscore: {}'.format(precision_micro, recall_micro, fscore_micro))



def save_model(model, model_filepath):
    '''
    Save model in model_filepath as py file

    INPUT:
    model               trained and optimzied model
    model_filepath      file path to save model


    '''   
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading Fraction of data for Optimization...\n    DATABASE: {}'.format(database_filepath))
        X_opt, Y_opt, category_names = load_data(database_filepath, 1)
        X_train_opt, X_test_opt, Y_train_opt, Y_test_opt = train_test_split(X_opt, Y_opt, test_size=0.2)
        
        print('Building and Optimizing model...')
        model, threshold_list = build_optimize_model(X_train_opt, Y_train_opt, X_test_opt, Y_test_opt)

        print('Loading data for Training...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath, 0)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Training model...')
        model = train_model(X_train, Y_train, model)

        #print('Training model...')
        #model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, threshold_list, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
                'as the first argument and the filepath of the pickle file to '\
                'save the model to as the second argument. \n\nExample: python '\
                'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()