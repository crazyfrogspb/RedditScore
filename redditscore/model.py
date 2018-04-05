"""
RedditTrainer: train and tune Reddit-based models
Author: Evgenii Nikitin <e.nikitin@nyu.edu>
"""

from abc import ABCMeta, abstractmethod
from itertools import product
import json
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import check_cv, PredefinedSplit, train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
try:
    import keras
except ModuleNotFoundError:
    print('keras could not be imported, some features will be unavailable')


def word_ngrams(tokens, ngram_range):
    """
    Extract ngrams from the tokenized sequence

    Parameters
    ----------
    tokens: iterable
        Sequence of tokens

    ngram_range: tuple (min_n, max_n), default: (1,1)
                The lower and upper boundary of the range of n-values for different n-grams to be extracted

    Returns
    ----------
    tokens: iterable
        Sequence of tokens with generated ngrams
    """
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

        for num in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - num + 1):
                tokens_append(space_join(original_tokens[i: i + num]))
    return tokens


def build_analyzer(ngram_range):
    """
    Build analyzer for vectorizers

    Parameters
    ----------
    ngram_range: tuple (min_n, max_n), default: (1,1)
                The lower and upper boundary of the range of n-values for different n-grams to be extracted
    """
    return lambda doc: word_ngrams(doc, ngram_range)


class RedditModel(metaclass=ABCMeta):
    def __init__(self, random_state=24):
        """
        Sklearn-style wrapper for the different architectures of the Reddit-based models

        random_state: int, default: 24
            Random seed
        """
        self.random_state = random_state
        np.random.seed(random_state)

        self.model_ = None
        self.model_type= None
        self.cv_split_ = None
        self.params = None

    def cv_score(self, X, y, cv=0.2, scoring='accuracy'):
        """
        Calculate validation score

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels

        cv: float, int, cross-validation generator or an iterable
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            -float, to use holdout set of this size
            -None, to use the default 3-fold cross validation,
            -integer, to specify the number of folds in a StratifiedKFold,
            -An object to be used as a cross-validation generator.
            -An iterable yielding train, test splits.

        scoring : string, callable or None, optional, default: 'accuracy'
            A string (see sklearn model evaluation documentation) or a scorer callable object

        Returns
        ----------
        Average value of the validation metrics
        """
        if isinstance(cv, float):
            train_ind, __ = train_test_split(np.arange(0, X.shape[0]))
            test_fold = np.zeros((X.shape[0], ))
            test_fold[train_ind] = -1
            self.cv_split_ = PredefinedSplit(test_fold)
        else:
            self.cv_split_ = check_cv(cv, y=y, classifier=True)

        return cross_val_score(self.model_, X, y, cv=self.cv_split_, scoring=scoring)

    def tune_params(self, X, y, param_grid=None, verbose=True, cv=0.2, scoring='accuracy'):
        """
        Find the best values of hyperparameters using chosen validation scheme

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels

        param_grid: dict, default: None
            Dictionary with parameters names as keys and lists of parameter settings
            If None, loads deafult values

        verbose: bool, default: True
            Whether to print scores after fitting each model

        cv: float, int, cross-validation generator or an iterable
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
            -float, to use holdout set of this size
            -None, to use the default 3-fold cross validation,
            -integer, to specify the number of folds in a StratifiedKFold,
            -An object to be used as a cross-validation generator.
            -An iterable yielding train, test splits.

        scoring : string, callable or None, optional, default: 'accuracy'
            A string (see sklearn model evaluation documentation) or a scorer callable object

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
                param_grid = json.load(f)[self.model_type]

        best_pars = None
        best_value = -1000000.0

        if not isinstance(param_grid, list):
            param_grid = [param_grid]

        for param_combination in param_grid:
            items = sorted(param_combination.items())
            keys, values = zip(*items)
            
            for v in product(*values):
                params = dict(zip(keys, v))
                self.set_params(**params)
                if verbose:
                    print('Now fitting model for {}'.format(params))
                score = np.mean(self.cv_score(X, y, cv, scoring))
                if verbose:
                    print('{}: {}'.format(scoring, score))
                if score > best_value:
                    best_pars = params
                    best_value = score
        if verbose:
            print('Best {}: {} for {}'.format(scoring, best_value, best_pars))
        return best_pars, best_value

    def fit(self, X, y):
        """
        Fit model

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels
        """
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the most likely label

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels

        Returns
        ----------

        array, shape (n_samples, )
            Predicted class labels
        """
        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Predict the most likely label

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels

        Returns
        ----------

        array, shape (n_samples, num_classes)
            Predicted class probabilities
        """
        return self.model_.predict_proba(X)

    def get_params(self, deep=None):
        """
        Get parameters of the model
        """
        return self.params

    @abstractmethod
    def set_params(self):
        pass


class SklearnModel(RedditModel):
    def __init__(self, model_type='multinomial', alpha=1.0, fit_prior=True, class_prior=None,
                 ngram_range=(1, 1), tfidf=True, C=1.0, kernel='linear', probability=True, **kwargs):
        """
        Naive Bayes classifier for multinomial and Bernoulli models with or without tf-idf re-weighting

        Parameters
        ----------
        multi_model: bool, default: True
            If True, use Multinomial model, if False, use Bernoulli model

        alpha: float, default: 1.0
            Additive (Laplace/Lidstone) smoothing parameter

        fit_prior: bool, default: True
            Whether to learn class prior probabilities or not. If false, a uniform prior will be used.

        class_prior: array-like, size (n_classes,), default: None
            Prior probabilities of the classes

        ngram_range: tuple (min_n, max_n), default: (1,1)
            The lower and upper boundary of the range of n-values for different n-grams to be extracted

        tfidf: bool, default: True
            Whether to use tf-idf re-weighting
        """
        super().__init__(**kwargs)
        if model_type in ['multinomial', 'bernoulli']:
            self.params = {'alpha': alpha, 'fit_prior': fit_prior, 'class_prior': class_prior, 'ngram_range': ngram_range,
                           'tfidf': tfidf}
        elif model_type == 'svm':
            self.params = {'C': C, 'kernel': kernel, 'probability': probability, 'ngram_range': ngram_range, 'tfidf': tfidf}
        else:
            raise ValueError('{} is not supported yet'.format(model_type))

        self.model_type = model_type
        self.set_params(**self.params)
        

    def set_params(self, **params):
        """
        Set the parameters of the model.
        """
        self.params.update(params)
        if self.model_type == 'multinomial':
            model = MultinomialNB(alpha=self.params['alpha'],
                                  fit_prior=self.params['fit_prior'], class_prior=self.params['class_prior'])
        elif self.model_type == 'bernoulli':
            model = BernoulliNB(alpha=self.params['alpha'],
                                fit_prior=self.params['fit_prior'], class_prior=self.params['class_prior'])
        elif self.model_type == 'svm':
            model = SVC(C=self.params['C'], kernel=self.params['kernel'], probability=self.params['probability'])

        if self.params['tfidf']:
            vectorizer = TfidfVectorizer(analyzer=build_analyzer(self.params['ngram_range']))
        else:
            vectorizer = CountVectorizer(analyzer=build_analyzer(self.params['ngram_range']))

        self.model_ = Pipeline([('vectorizer', vectorizer), ('model', model)])

        return self

class FasttextModel(RedditModel):
    pass

class MLPModel(RedditModel):
    pass

class CNNModel(RedditModel):
    pass

class LSTMModel(RedditModel):
    pass