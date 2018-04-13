# -*- coding: utf-8 -*-
"""
Generic RedditModel class for specific models to inherit

Author: Evgenii Nikitin <e.nikitin@nyu.edu>

Part of https://github.com/crazyfrogspb/RedditScore project

Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""

import json
import os
from abc import ABCMeta
from itertools import product

import numpy as np
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import (PredefinedSplit, check_cv,
                                     cross_val_score, train_test_split)


def word_ngrams(tokens, ngram_range):
    # Extract ngrams from the tokenized sequence
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


class RedditModel(metaclass=ABCMeta):
    """
    Sklearn-style wrapper for the different architectures

    random_state: int, optional
        Random seed
    """

    def __init__(self, random_state=24):
        self.random_state = random_state
        self.model_type = None
        self._model = None
        self._cv_split = None
        self.params = None
        self._classes = None

    def cv_score(self, X, y, cv=0.2, scoring='accuracy'):
        """
        Calculate validation score

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels

        cv: float, int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - float, to use holdout set of this size
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a StratifiedKFold,
            - An object to be used as a cross-validation generator.
            - An iterable yielding train, test splits.

        scoring : string, callable or None, optional, optional
            A string (see sklearn model evaluation documentation) or a scorer callable object

        Returns
        ----------
        float
            Average value of the validation metrics
        """
        self._classes = sorted(np.unique(y))
        np.random.seed(self.random_state)
        if isinstance(cv, float):
            train_ind, __ = train_test_split(np.arange(0, X.shape[0]))
            test_fold = np.zeros((X.shape[0], ))
            test_fold[train_ind] = -1
            self._cv_split = PredefinedSplit(test_fold)
        else:
            self._cv_split = check_cv(cv, y=y, classifier=True)

        if scoring == 'neg_log_loss':
            scoring = make_scorer(log_loss, labels=self._classes,
                                  greater_is_better=False, needs_proba=True)
        return cross_val_score(self._model, X, y, cv=self._cv_split,
                               scoring=scoring)

    def tune_params(self, X, y, param_grid=None,
                    verbose=False, cv=0.2, scoring='accuracy', refit=False):
        """
        Find the best values of hyperparameters using chosen validation scheme

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels

        param_grid: dict, optional
            Dictionary with parameters names as keys and lists of parameter settings
            If None, loads deafult values from JSON file

        verbose: bool, optional
            If True, print scores after fitting each model

        cv: float, int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - float, to use holdout set of this size
            - None, to use the default 3-fold cross validation,
            - integer, to specify the number of folds in a StratifiedKFold,
            - An object to be used as a cross-validation generator.
            - An iterable yielding train, test splits.

        scoring : string, callable or None, optional
            A string (see sklearn model evaluation documentation) or a scorer callable object

        refit: boolean, optional
            If True, refit model with the best found parameters

        Returns
        ----------
        best_pars: dict
            Dictionary with the best combination of parameters
        best_value: float
            Best value of the chosen metric
        """
        self._classes = sorted(np.unique(y))
        if param_grid is None:
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                os.path.join('..', 'data', 'model_pars.json'))
            with open(file) as f:
                param_grid = json.load(f)[
                    self.model_type]

        if 'step0' not in param_grid:
            param_grid_temp = {'step0': param_grid}
            param_grid = param_grid_temp

        for step in range(len(param_grid)):
            best_pars = None
            best_value = -1000000.0
            if verbose:
                print('Fitting step {}'.format(step))

            try:
                current_grid = param_grid['step{}'.format(step)]
            except KeyError:
                raise KeyError('Step{} is not in the grid'.format(step))

            if not isinstance(current_grid, list):
                current_grid = [current_grid]

            for param_combination in current_grid:
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

            self.set_params(**best_pars)

        if verbose:
            print('Best {}: {} for {}'.format(scoring, best_value, best_pars))

        if refit:
            self.set_params(**best_pars)
            self.fit(X, y)

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
        self._classes = sorted(np.unique(y))
        self._model.fit(X, y)
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

        X = np.array(X)
        return self._model.predict(X)

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
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return self._model.predict_proba(X)

    def get_params(self, deep=None):
        """
        Get parameters of the model
        """
        return self.params

    def set_params(self, **params):
        """
        Set parameters of the model
        """
        self.params.update(params)
