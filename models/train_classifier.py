# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    '''
    Load the data from the sql lite database from path mentioned in the 
    database_filepath argument. Split the data into X and Y for the model and 
    return the dfs
    '''
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(con = engine, table_name = 'Disaster_Response_Table')
    X = df['message']
    Y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    return X, Y


def tokenize(text):
    '''
    Function to tokenize the text passed as argument
    '''
    
    # convert to lower case
    text = text.lower()
    
    # remove special characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    ## lemmatization
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(clean_token)
    
    # return tokens
    return clean_tokens


def build_model(X_train, Y_train):
    '''
    Function to build the model. Pipeline is created for the pre-processing
    steps fillowed by training the pipeline over a grid search to find the
    best model
    '''
    
    
    # build pre-processing and ML pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])
    
    # initialize parameters for grid search 

    parameters = {'vect__max_df' : [0.5, 1.0],
                  'vect__max_features' : [None, 5000],
                  'tfidf__use_idf' : [True, False],
                  'clf__estimator__min_samples_leaf' : [1, 3],
                  'clf__estimator__n_estimators' : [10, 20]}

    
    # initialize grid search
    cv = GridSearchCV(pipeline, param_grid = parameters,
                      verbose = 2, cv = 2)
    
    
    # train the pipeline (with model)
    cv.fit(X_train, Y_train)
    
    # return best model
    return cv.best_estimator_


def evaluate_model(model, X_test, Y_test):
    '''
    Evaluate the model passed as the parameter by first making the predictions
    on X_test and evaluating the results on Y_test. The function prints 
    evaluation metrics on all the labels
    '''
    
    
    # make predictions on test data
    Y_test_pred = pd.DataFrame(model.predict(X_test))
    
    # print the model performance
    for col_index in range(0, Y_test.shape[1]):
        print(f'---- Classification Report for {Y_test.columns[col_index]} ----\n', 
              classification_report(Y_test.loc[:, Y_test.columns[col_index]], 
                                    Y_test_pred.loc[:, Y_test_pred.columns[col_index]],
                                    labels = Y_test.loc[:, Y_test.columns[col_index]].unique()),
              '------------------------------------------------------\n')


def save_model(model, model_filepath):
    '''
    Save the model passed as argument to a pickle file to the 
    model_filepath location
    '''
    
    # save the model to pickle file
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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