"""
RedditTrainer: train and tune Reddit-based models
Author: Evgenii Nikitin <e.nikitin@nyu.edu>
"""

import pandas as pd
from sklearn.model_selection import check_cv, PredefinedSplit, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import keras
from .redditscore import SpacyTokenizer


class RedditModel(object):
    def __init__(self, model, reddit_data=None, body_col="body", subreddit_col="subreddit",
                 val_type="holdout", cv=None, val_size=0.2, tokenizer=SpacyTokenizer()):
        """
        Sklearn-style wrapper for the different architectures of the Reddit-based models

        Parameters
        ----------
        model: sklearn-style model
            Model object that will be trained

        reddit_data: pandas DataFrame, default: None
            DataFrame with the training data

        body_col: str, default: "body"
            Name of the column with texts

        subreddit_col: str, default: "subreddit"
            Name of the column with the labels

        val_type: str, default: "holdout"
            Validation type for parameters tuning and/or assesing model quality
                None: no validation
                "holdout": using holdout validation set
                "cv": using cross-validation

        cv: int, cross-validation generator or an iterable
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                An object to be used as a cross-validation generator.
                An iterable yielding train, test splits.

        val_size: float, default: 0.2
            Size of the validation set when using "holdout" validation strategy

        tokenizer: tokenizer object with ".tokenize" method, default: SpacyTokenizer()
            Tokenizer for splitting texts into tokens
        """
        if isinstance(reddit_data, str):
            try:
                self.df = pd.read_csv(reddit_file)
            except FileNotFoundError as e:
                raise Exception('File {} does not exist'.format(reddit_data))
        elif isinstance(reddit_data, pd.DataFrame):
            self.df = reddit_data
        else:
            raise TypeError(
                'Parameter reddit_data should contain either path to CSV file with the data or pandas DataFrame object')

        if body_col in self.df.columns:
            self.body_col = body_col
        else:
            raise ValueError('Column {} is not in the dataframe'.format{body_col})

        if subreddit_col in self.df.columns:
            self.subreddit_col = subreddit_col
        else:
            raise ValueError('Column {} is not in the dataframe'.format(subreddit_col))

        if val_type not in [None, 'holdout', 'cv']:
            raise ValueError("val_type should be either None, 'holdout', or 'cv'")
        elif val_type = 'cv':
            self.cv_split = check_cv(cv, y=self.df[self.subreddit_col], classifier=True)
        elif val_type == 'holdout':
            self.cv_split = None
        self.val_type = val_type

    def fit(self, documents, labels):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


class MNBModel(RedditModel):
    pass


class SklearnClassifierWrapper(object):
    def __init__(self, model, tfidf=False, ngram_n=1):
        """
        Classifier made up of a pipeline with a count vectorizer + given model
        :param model: a sklearn-like classifier (with fit, predict and predict_proba)
        :param tfidf: if True wil use TfidfVectorizer, otherwise CountVectorizer; defaults to False
        """
        vectorizer_class = TfidfVectorizer if tfidf else CountVectorizer
        vectorizer = vectorizer_class(
            preprocessor=lambda x: map(str, x),
            tokenizer=lambda x: x,
            ngram_range=(1, ngram_n))

        self.params = {'tfidf': tfidf, 'ngram_n': ngram_n}
        self.clf = Pipeline([('vectorizer', vectorizer), ('model', model)])
        self.name = "SklearnClassifierWrapper(tfidf=%s)" % tfidf

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

    def get_params(self, deep=None):
        return self.params

    def __str__(self):
        return self.name
