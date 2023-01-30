import sys
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine
from sklearn.neighbors import KNeighborsClassifier

def load_data(database_filepath):
    """
    input : database filepath  
    
    output :X - messages (input variable) 
            y - categories of the messages (target variable)
            category_names - all category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    input : text-message
    output :clean_tokens-tokenized text
    """
    
    #removing symbols that are not usefull for our findings
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
   #tokenize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #remove capitals
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
   
    return clean_tokens


def build_model():
    """
    output: cv = ML model pipeline after performing grid search with KNeighborsClassifier
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    print(pipeline.get_params().keys())
    k_range = list(range(1, 31))
    parameters = {'clf__estimator__n_neighbors':k_range}

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1,
                               cv=2, return_train_score=True, verbose=3)
    
    return pipeline



    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    input:  model - pipeline
            X_test - test messages
            y_test - target categories for test messages
            category_names - category names
    
    output: print scores for each output category of the dataset.
    """
    
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))
    

def save_model(model, model_filepath):
    """
    input: model - pipeline
    model_filepath - location to save the model
    """
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