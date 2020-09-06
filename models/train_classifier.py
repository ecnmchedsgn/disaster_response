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

# import packages to save model
import pickle

def load_data(database_filepath):
    '''
    Input:
        database_filepath   filepath to SQL database

    Output:
        X               (pandas Series) messages to categorize (str in English)
        Y               (pandas dataframe) 36 category columns for the messages
        category_names  (list) 36 category names

    Description:

    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('categorized_messages', engine) 

    # include for fast tests, subsample dataset
    # df = df.sample(frac=0.05, replace=False, random_state=1)

    # Upsample to include more samples(rows in df) with non-zero entries
    df_upsampled = upsample_df(df)

    X = df_upsampled['message']
    Y = df_upsampled.iloc[:,4:]


    category_names = list(Y.columns.values)

    return X, Y, category_names

def upsample_df(df):
    '''
    '''
    # Separate majority and minority classes
    df_majority = df.loc[df.iloc[:,4:].sum(axis=1) == 0]
    df_minority = df.loc[df.iloc[:,4:].sum(axis=1) > 0]
    
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=df_minority.shape[0]*5,    # to increase size 5 times
                                     random_state=123) # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
    return df_upsampled

def build_model():
    '''
    '''
    # build pipeline using multilabel-multioutput(multiclass) classification
    pipeline_moc = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multi_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    return pipeline_moc


def tokenize(text):
    '''
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    '''
 
    # Downsample test dataset
    X_test_downsampled, Y_test_downsampled = down_sample(X_test, Y_test)

    # predict 
    y_pred = model.predict(X_test_downsampled)
    
    # print out results for each category
    # for each label, output classification report
    for ind in range(0,len(Y_test_downsampled.columns)):
    
        print('classification report for category: '+Y_test_downsampled.columns.values[ind])
        print(classification_report(Y_test_downsampled.loc[:,Y_test_downsampled.columns.values[ind]], y_pred[:,ind])) 
    
    
    # find overall classifier performance (macro average & treating multi-classifier as 3 labels)
    # the multi-label part
    precision = []
    recall = []
    fscore = []
    for ind in range(1,len(Y_test_downsampled.columns)):
    
        #print(ind)
        #print(y_test_downsampled.columns.values[ind])
        classify = precision_recall_fscore_support(Y_test_downsampled.loc[:,Y_test_downsampled.columns.values[ind]], 
                                                y_pred[:,ind], average='binary', zero_division=0)
        precision.append(classify[0])
        recall.append(classify[1])
        fscore.append(classify[2])
    
    # the multi-classification part
    # precision, for label 0, 1 and 2
    precision_mc = precision_recall_fscore_support(Y_test_downsampled.loc[:,Y_test_downsampled.columns.values[0]], y_pred[:,0],zero_division=0)[0]
    # recall,
    recall_mc = precision_recall_fscore_support(Y_test_downsampled.loc[:,Y_test_downsampled.columns.values[0]], y_pred[:,0],zero_division=0)[1]
    # f1
    fscore_mc = precision_recall_fscore_support(Y_test_downsampled.loc[:,Y_test_downsampled.columns.values[0]], y_pred[:,0],zero_division=0)[2]

    # macro average
    precision_macro = (sum(precision)+sum(precision_mc))/(len(precision)+3)
    recall_macro = (sum(recall)+sum(recall_mc))/(len(recall)+3)
    fscore_macro = (sum(fscore)+sum(fscore_mc))/(len(fscore)+3)
    print('Macro averaged precion: {}, recall: {}, fscore: {}'.format(precision_macro, recall_macro, fscore_macro))


    # find overall classifier performance (micro average)
    # multi-label part
    confusion_matrix_alllabels = multilabel_confusion_matrix(Y_test_downsampled.iloc[:,1:], y_pred[:,1:])
    TN, FN, FP, TP = 0, 0, 0, 0
    for ind in range(0,35):
        TN += confusion_matrix_alllabels[ind][0,0] # true negative
        FN += confusion_matrix_alllabels[ind][1,0] #false negative
        FP += confusion_matrix_alllabels[ind][0,1] #false postitive
        TP += confusion_matrix_alllabels[ind][1,1] #true positive
    
    # multi-class part
    confusion_mc = confusion_matrix(Y_test_downsampled.loc[:,Y_test_downsampled.columns.values[0]], y_pred[:,0])
    # treat each class as a seprate label
    TP += confusion_mc[0,0]
    TN += confusion_mc[1,1] + confusion_mc[1,2] + confusion_mc[2,1]+ confusion_mc[2,2] 
    FP += confusion_mc[0,1] + confusion_mc[0,2]
    FN += confusion_mc[1,0] + confusion_mc[2,0]


    TP += confusion_mc[1,1]
    TN += confusion_mc[0,0] + confusion_mc[0,2] + confusion_mc[2,0]+ confusion_mc[2,2] 
    FP += confusion_mc[1,0] + confusion_mc[1,2]
    FN += confusion_mc[0,1] + confusion_mc[2,1]

    TP += confusion_mc[2,2]
    TN += confusion_mc[0,0] + confusion_mc[0,1] + confusion_mc[1,0]+ confusion_mc[1,1] 
    FP += confusion_mc[2,0] + confusion_mc[2,1]
    FN += confusion_mc[0,2] + confusion_mc[1,2]

    precision_micro = TP/(TP+FP)
    recall_micro = TP/(TP+FN)
    fscore_micro = 2*precision_micro*recall_micro/(recall_micro+precision_micro)
    print('Micro averaged precion: {}, recall: {}, fscore: {}'.format(precision_micro, recall_micro, fscore_micro))


def down_sample(X_test, Y_test):
    '''
    '''
    # downsample  test dataset
    Y_minority_test = Y_test.loc[Y_test.sum(axis=1) > 0]
    X_minority_test = X_test.loc[Y_test.sum(axis=1) > 0]

    Y_majority_test = Y_test.loc[Y_test.sum(axis=1) == 0] 
    X_majority_test = X_test.loc[Y_test.sum(axis=1) == 0]


    df_test_majority = Y_majority_test.join(X_majority_test)
    df_test_minority = Y_minority_test.join(X_minority_test)


    # downsample minority class
    df_test_minority_downsampled = resample(df_test_minority, 
                                    replace=False,     # sample method
                                    n_samples=df_test_majority.shape[0]*3, # this gives roughly 75% non-zero entries     
                                    random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    df_test_downsampled = pd.concat([df_test_majority, df_test_minority_downsampled])

    # Output X and Y variables
    X_test_downsampled = df_test_downsampled['message']
    Y_test_downsampled = df_test_downsampled.iloc[:,:-1] #excluse message column

    return X_test_downsampled, Y_test_downsampled

def save_model(model, model_filepath):
    '''
    '''   
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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