# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input:
        messages_filepath       filepath for messages csv
        categories_filepath     filepath for categories csv

    Output:
        df  (pandas dataframe) merged dataframe with following columns
        id	| message	| original |	genre	| categories

    Description:
        Loads messages and cateogires dataset and output combined dataset. 
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on = 'id')

    return df

def clean_data(df):
    '''
    Description:
    Clean data in the following steps
        Split categories into separate category columns. Rename columns of categories with new column names.
        Convert category values to just numbers 0 or 1.
        Replace categories column in df with new category columns.
        Remove duplicates
    
    Input: 
        df (pandas dataframe) dataset containing message and 1 raw category colum 
    Output:
        df (pandas dataframe) dataset containing message and 36 cateogry columns with 0 or 1 values, with duplicates removed.
    '''
    ## create a dataframe of the 36 individual category columns 
    # split on the ; character
    categories = df['categories'].str.split(";",expand=True)

    # select the first row of the categories dataframe, # extract a list of new column names for categories
    row = categories.iloc[0]

    # takes everything up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    ## convert category values to 0 or 1 
    # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].str[-1:]
    
        # convert from strings to numerical
        categories[column] = pd.to_numeric(categories[column]).astype(np.int64)

        # if related columns, then assume 2 as 1's
        if column == 'related':
            # assume value of 2 is same as 1
            categories[column] = categories[column].apply(lambda x: 1 if x==2 else x)


    ## concatenate cateogires dataframe with original dataframe df
    # drop the original categories column from `df`
    df = df.drop(columns = 'categories')

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates(keep = 'first')
    # check number of duplicates is zero
    assert df[df.duplicated()].shape[0] == 0, 'Oops, some duplicates are not removed'

    # check that each column/label has 2 classification labels
    cols_drop = []
    for column in df.columns[4:]:
        if df[column].nunique() != 2:
            cols_drop.append(column)
    
    # drop columns with only 1 classification label
    for col in cols_drop:
        print('Column {} has only 1 label and is dropped'.format(col))
    df = df.drop(columns = cols_drop) 
    
    return df


def save_data(df, database_filename):
    '''
    Input: 
        df                      (pandas dataframe) cleaned dataset containing messages and 36 category columns
        database_filename       path to create SQL database
    '''
    # create SQL engine/database with the specified database_filename
    # Since the file is saved locally, use sqlite://<nohostname>/<path>,  where <path> is relative
    engine = create_engine('sqlite:///'+database_filename)

    # save dataframe as TABLE named categorized_messages in database
    df.to_sql('categorized_messages', engine, index=False, if_exists = 'replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()