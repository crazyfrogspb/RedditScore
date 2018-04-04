"""
RedditTrainer: train and tune Reddit-based models
Author: Evgenii Nikitin <e.nikitin@nyu.edu>
"""

import pandas as pd
from sklearn.model_selection import check_cv, PredefinedSplit, train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import keras
from .tokenizer import CrazyTokenizer
from sklearn.pipeline import Pipeline
import numpy as np
from abc import ABCMeta, abstractmethod
from itertools import product
import json
import os


class RedditModel(metaclass=ABCMeta):
    def __init__(self, reddit_data, body_col="body", subreddit_col="subreddit",
                 val_type="holdout", cv=None, val_size=0.2, tokenizer=CrazyTokenizer(),
                 random_state=24):
        """
        Sklearn-style wrapper for the different architectures of the Reddit-based models

        Parameters
        ----------
        reddit_data: pandas DataFrame or str
            DataFrame with the training data or path to CSV file with data

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

        tokenizer: tokenizer object with ".tokenize" method, default: CrazyTokenizer()
            Tokenizer for splitting texts into tokens

        random_state: int, default: 24
            Random seed
        """
        self.random_state = random_state
        np.random.seed(random_state)

        if isinstance(reddit_data, str):
            try:
                self.df = pd.read_csv(reddit_data)
            except FileNotFoundError as e:
                raise Exception('File {} does not exist'.format(reddit_data))
        elif isinstance(reddit_data, pd.DataFrame):
            self.df = reddit_data.reset_index(drop=True)
        else:
            raise TypeError(
                'Parameter reddit_data should contain either path to CSV file with the data or pandas DataFrame object')

        if body_col in self.df.columns:
            self.body_col = body_col
        else:
            raise ValueError('Column {} is not in the dataframe'.format(body_col))

        if subreddit_col in self.df.columns:
            self.subreddit_col = subreddit_col
        else:
            raise ValueError('Column {} is not in the dataframe'.format(subreddit_col))

        if val_type not in [None, 'holdout', 'cv']:
            raise ValueError("val_type should be either None, 'holdout', or 'cv'")
        elif val_type == 'cv':
            self.cv_split = check_cv(cv, y=self.df[self.subreddit_col], classifier=True)
        elif val_type == 'holdout':
            train_ind, __ = train_test_split(self.df.index.values)
            test_fold = np.zeros((self.df.shape[0], ))
            test_fold[train_ind] = -1
            self.cv_split = PredefinedSplit(test_fold)

        self.tokenizer_ = tokenizer
        self.model_ = None
        self.model_name_ = None

        self.df['redditscore_tokens'] = self.df[self.body_col].apply(self.tokenizer_.tokenize)

    def cv_score(self, scoring='accuracy'):
        """
        Calculate validation score

        Parameters
        ----------
        scoring : string, callable or None, optional, default: 'accuracy'
            A string (see sklearn model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).

        Returns
        ----------
        Average value of thevalidation metrics
        """
        return cross_val_score(self.model_, X=self.df['redditscore_tokens'],
                               y=self.df[self.subreddit_col], cv=self.cv_split, scoring=scoring)

    def tune_params(self, param_grid=None, verbose=True, scoring='accuracy'):
        """
        Find the best values of hyperparameters using chosen validation scheme

        Parameters
        ----------
        param_grid: dict, default: None
            Dictionary with parameters names as keys and lists of parameter settings
            If None, loads deafult values

        verbose: bool, default: True
            Whether to print scores after fitting each model

        scoring : string, callable or None, optional, default: 'accuracy'
            A string (see sklearn model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).

        Returns
        ----------
        best_pars: dict
            Dictionary with the best combination of parameters
        best_value: float
            Best value of the chosen metric
        """
        if param_grid is None:
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                os.path.join('data', 'model_pars.json'))
            with open(file) as f:
                param_grid = json.load(f)[self.model_name_]
        items = sorted(param_grid.items())
        keys, values = zip(*items)
        best_pars = None
        best_value = -1000000.0
        for v in product(*values):
            params = dict(zip(keys, v))
            self.set_params(**params)
            if verbose:
                print('Now fitting model for {}'.format(params))
            score = np.mean(self.cv_score(scoring=scoring))
            if verbose:
                print('Accuracy: {}'.format(score))
            if score > best_value:
                best_pars = params
                best_acc = score
        if verbose:
            print('Best accuracy: {} for {}'.format(best_acc, best_pars))
        return best_pars, best_value

    def fit(self, X=None, y=None):
        """
        Fit model

        Parameters
        ----------
        X: iterable, default: None
            Sequence of training documents, if None, then use preloaded data
        y: iterable, deafult: None
            Sequence of training labels, if None, then use preloaded data

        Returns
        ----------
        self
        """
        if X is None:
            X = self.df['redditscore_tokens']
        if y is None:
            y = self.df[self.subreddit_col]
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the most likely label

        Parameters
        ----------
        X: iterable
            Sequence of tokenized or non-tokenized documents

        Returns
        ----------
        np.array(shape: (X.shape[0], ))
            Predicted labels
        """
        if isinstance(X[0], list):
            return self.model_.predict(X)
        else:
            tokenized_documents = [self.tokenizer_.tokenize(doc) for doc in X]
            return self.model_.predict(tokenized_documents)

    def predict_proba(self, X):
        if isinstance(X[0], list):
            return self.model_.predict_proba(X)
        else:
            tokenized_documents = [self.tokenizer_.tokenize(doc) for doc in X]
            return self.model_.predict_proba(tokenized_documents)

    def get_params(self, deep=None):
        return self.params

    @abstractmethod
    def set_params(self):
        pass


class BayesModel(RedditModel):
    def __init__(self, reddit_data,
                 multi_model=True, alpha=1.0, fit_prior=True, class_prior=None, ngram_range=(1, 1), tfidf=True, **kwargs):
        super().__init__(reddit_data, **kwargs)
        self.params = {'alpha': alpha, 'fit_prior': fit_prior, 'class_prior': class_prior, 'ngram_range': ngram_range,
                       'tfidf': tfidf, 'multi_model': multi_model}
        self.set_params(**self.params)
        self.model_name_ = 'BayesModel'

    def build_analyzer(self, ngram_range):
        return lambda doc: self._word_ngrams(doc, ngram_range)

    def _word_ngrams(self, tokens, ngram_range):
        min_n, max_n = ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                           min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))
        return tokens

    def set_params(self, **params):
        self.params.update(params)
        if self.params['multi_model']:
            model = MultinomialNB(alpha=self.params['alpha'],
                                  fit_prior=self.params['fit_prior'], class_prior=self.params['class_prior'])
        else:
            model = BernoulliNB(alpha=self.params['alpha'],
                                fit_prior=self.params['fit_prior'], class_prior=self.params['class_prior'])
        if self.params['tfidf']:
            vectorizer = TfidfVectorizer(analyzer=self.build_analyzer(self.params['ngram_range']))
        else:
            vectorizer = CountVectorizer(analyzer=self.build_analyzer(self.params['ngram_range']))

        self.model_ = Pipeline([('vectorizer', vectorizer), ('model', model)])

        return self
