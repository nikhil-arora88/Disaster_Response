# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load the data sets, merge them and return the merged data set
    File paths are passed as arguments to the function
    '''
   
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on = 'id', how = 'outer')
    
    # return merged dataset
    return df
    


def clean_data(df):
    '''
    Function for the clearning the dataframe passed as the argument
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    
    # use this row to extract a list of new column names for categories
    category_colnames = row.map(lambda x: str(x)[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        
        # convert column from string to numeric
        categories.replace({column : {'2' : '1'}}, inplace = True)
        categories[column] = categories[column].astype(int)
        
        if (categories[column].nunique() < 2):
            categories.drop(columns = column, inplace = True)
        
    
    # drop the original categories column from `df`
    df.drop(columns = 'categories', inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    df.drop_duplicates(subset = 'id', inplace = True)
    
    # return cleaned df
    return df



def save_data(df, database_filename):
    '''
    Function to save the df dataframe to a sql lite database with 
    path mentioned in database_filename argument
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Disaster_Response_Table', engine, index = False, if_exists = 'replace')  



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