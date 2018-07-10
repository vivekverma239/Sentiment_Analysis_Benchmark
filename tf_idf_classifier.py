import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from utils import DataProcessor
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from nbsvm import NBSVM
import string



class Model(object):
    def __init__(self,data_file,seperator=','):

        data_processor = DataProcessor(data_file,seperator=seperator,raw_data=True)
        self.data , self.labels    = data_processor.get_training_data(raw_text=True)

        self.X_train = self.data
        self.y_train = self.labels


        test_data, test_labels = data_processor.process_test_file( '../data/imdb/test.csv',contains_label=True,header=0)

        print('Running Naive Bayes...')
        pipeline, parameters =self.get_naive_bayes_model()
        grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=4, verbose=10)
        grid_search_tune.fit(self.X_train, self.y_train)
        print("Best parameters set:")
        self.best_estimator_ =  grid_search_tune.best_estimator_
        print(grid_search_tune.best_score_)
        self.calculate_metric( test_data,test_labels)
        print('#'*80)

        print('Running Linear SVM...')
        pipeline, parameters = self.get_linear_svm_model()
        grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=4, verbose=10)
        grid_search_tune.fit(self.X_train, self.y_train)
        print("Best parameters set:")
        self.best_estimator_ =  grid_search_tune.best_estimator_
        print(grid_search_tune.best_score_)
        self.calculate_metric( test_data,test_labels)
        print('#'*80)

        print('Running Non Linear SVM...')
        pipeline, parameters = self.get_non_linear_svm_model()
        grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=4, verbose=10)
        grid_search_tune.fit(self.X_train, self.y_train)
        print("Best parameters set:")
        self.best_estimator_ =  grid_search_tune.best_estimator_
        print(grid_search_tune.best_score_)
        self.calculate_metric()
        print('#'*80)

        print('Running Naive Bayes SVM...')
        pipeline, parameters = self.get_nbsvm_model()
        grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=6, verbose=10)
        grid_search_tune.fit(self.X_train, self.y_train)
        print("Best parameters set:")
        self.best_estimator_ =  grid_search_tune.best_estimator_
        print(grid_search_tune.best_score_)
        self.calculate_metric( test_data,test_labels)
        print('#'*80)

    def get_naive_bayes_model(self):
        token_pattern = r'\w+|[%s]' % string.punctuation
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(token_pattern=token_pattern)),
            ('clf', MultinomialNB(fit_prior=True))
        ])
        parameters = {
            # 'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            # 'tfidf__max_features': (100,500,1000,5000, 10000),
            'clf__alpha':(1e-3,1e-2,1e-1,1)
        }

        return pipeline, parameters



    def get_linear_svm_model(self):
        token_pattern = r'\w+|[%s]' % string.punctuation
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(token_pattern=token_pattern)),
            ('clf', LinearSVC( max_iter=1000))
        ])

        parameters = {
            # 'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            # 'tfidf__max_features': (100,500,1000,5000, 10000),
            # 'clf__penalty':('l1','l2'),
            'clf__loss':('squared_hinge','hinge'),
            'clf__C':(1,5,10),
        }
        return pipeline, parameters

    def get_nbsvm_model(self):
        token_pattern = r'\w+|[%s]' % string.punctuation
        pipeline = Pipeline([
            ('tfidf', CountVectorizer(binary=True,token_pattern=token_pattern)),
            ('clf', NBSVM())
        ])

        parameters = {
            # 'tfidf__max_df': (0.25, 0.5, 0.75,1.0),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            # 'tfidf__max_features': (100,500,1000,10000),
            # 'clf__alpha':(1e-3,1e-2,1e-1,1),
            # 'clf__loss':('squared_hinge','hinge'),
            # 'clf__C':(1,5,10),
        }
        return pipeline, parameters

    def get_non_linear_svm_model(self):


        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SVC())
        ])

        parameters = {
            # 'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            # 'tfidf__max_features': (100,500,1000,5000, 10000),
            'clf__kernel':('rbf','sigmoid'),
            'clf__C':(1,5,10),
        }
        return pipeline, parameters

    def calculate_metric(self, test_data,test_labels):
        print('Test Metrics:')
        predicted = self.best_estimator_.predict(test_data)
        print('Accuracy:', np.mean(predicted == test_labels))
        print(classification_report(test_labels,predicted))
        # print('Test Score:', np.mean(predicted == self.y_test))

if __name__ == '__main__':
    data_path = '../data/imdb/train.csv'
    temp = Model(data_path)
