# -*- coding: utf-8 -*-
"""
Generic RedditModel class for specific models to inherit

Author: Evgenii Nikitin <e.nikitin@nyu.edu>

Part of https://github.com/crazyfrogspb/RedditScore project

Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""

import collections
import json
import os
import warnings
from abc import ABCMeta
from collections import Sequence
from itertools import product

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from adjustText import adjust_text
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.manifold import TSNE
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import (PredefinedSplit, check_cv,
                                     cross_val_score, train_test_split)

DEFAULT_LINKAGE_PARS = {'method': 'average', 'metric': 'cosine',
                        'optimal_ordering': True}
DEFAULT_DENDROGRAM_PARS = {'leaf_font_size': 20, 'max_d': 0.75,
                           'orientation': 'right', 'distance_sort': True}
DEFAULT_CLUSTERING_PARS = {'t': 0.75, 'criterion': 'distance'}
DEFAULT_TSNE_PARS = {'perplexity': 10.0, 'early_exaggeration': 30.0,
                     'learning_rate': 5.0, 'n_iter': 1000, 'method': 'exact',
                     'random_state': 24}
DEFAULT_LEGEND_PARS = {'loc': 'best', 'bbox_to_anchor': (1, 0.5),
                       'fancybox': True, 'shadow': True, 'labels': [],
                       'fontsize': 16}


def top_k_accuracy_score(y_true, y_pred, k=3, normalize=True):
    true_labels = list(y_pred.columns)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.values)
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_pred.shape
    idx = num_labels - k - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if true_labels.index(y_true[i]) in argsorted[i, idx + 1:]:
            counter += 1
    if normalize:
        return counter / num_obs
    else:
        return counter


def fancy_dendrogram(z, labels, **kwargs):
    # Function to plot fancy dendrograms
    # Taken from:
    # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = hac.dendrogram(z, labels=labels, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Class')
        plt.ylabel('Metric')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def word_ngrams(tokens, ngram_range, separator=' '):
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
        space_join = separator.join

        for num in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - num + 1):
                tokens_append(space_join(original_tokens[i: i + num]))
    return tokens


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


class RedditModel(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Sklearn-style wrapper for the different architectures

    Parameters
    ----------
    random_state : int, optional
        Random seed (the default is 24).

    Attributes
    ----------
    model_type : str
        Model type name
    model : model object
        Model object that is being fitted
    params : dict
        Dictionary with model parameters
    _classes : list
        List of class labels
    fitted : bool
        Indicates whether model was fitted
    class_embeddings : np.array, shape (num_classes, vector_size)
        Matrix with class embeddings
    random_state: int
        Random seed used for validation splits and for models
    """

    def __init__(self, random_state=24):
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        self.fitted = False
        self.class_embeddings = None
        self.params = {}

        np.random.seed(random_state)

    def cv_score(self, X, y, cv=0.2, scoring='accuracy', k=3):
        """Calculate validation score

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
            A string (see sklearn model evaluation documentation) or a scorer callable object or 'top_k_accuracy'

        k: int, optional
            k parameter for 'top_k_accuracy' scoring

        Returns
        ----------
        float
            Average value of the validation metrics
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if (not hasattr(y[0], '__array__') and isinstance(y[0], Sequence)
                and not isinstance(y[0], str)):
            raise ValueError(
                'Cross validation does not support multilabels yet')

        self.classes_ = sorted(np.unique(y))
        np.random.seed(self.random_state)
        if isinstance(cv, float):
            train_ind, __ = train_test_split(np.arange(0, len(X)),
                                             test_size=cv, shuffle=True,
                                             random_state=self.random_state)
            test_fold = np.zeros((len(X), ))
            test_fold[train_ind] = -1
            cv_split = PredefinedSplit(test_fold)
        else:
            cv_split = check_cv(cv, y=y, classifier=True)

        if scoring == 'neg_log_loss':
            scoring = make_scorer(log_loss, labels=self.classes_,
                                  greater_is_better=False, needs_proba=True)
        elif scoring == 'top_k_accuracy':
            scoring = make_scorer(top_k_accuracy_score, k=k,
                                  greater_is_better=True, needs_proba=True)
        return cross_val_score(self.model, X, y, cv=cv_split,
                               scoring=scoring)

    def tune_params(self, X, y, param_grid=None,
                    verbose=False, cv=0.2, scoring='accuracy', k=3, refit=False):
        """Find the best values of hyperparameters using chosen validation scheme

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels

        param_grid: dict, optional
            Dictionary with parameters names as keys and
            lists of parameter settings as values.
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
            A string (see sklearn model evaluation documentation) or a scorer callable object or 'top_k_accuracy'

        k: int, optional
            k parameter for 'top_k_accuracy' scoring

        refit: boolean, optional
            If True, refit model with the best found parameters

        Returns
        ----------
        best_pars: dict
            Dictionary with the best combination of parameters
        best_value: float
            Best value of the chosen metric
        """
        self.classes_ = sorted(np.unique(y))

        model_name = None
        if param_grid is None:
            model_type = self.model.__class__.__name__
            if model_type == 'FastTextClassifier':
                model_name = 'fasttext'
            elif model_type == 'Pipeline':
                model_type = self.model.named_steps['model'].__class__.__name__
                if model_type in ['SVC', 'SVR']:
                    model_name = 'SVM'
                elif model_type in ['BernoulliNB', 'MultinomialNB']:
                    model_name = 'bayes'
            if model_name is None:
                raise ValueError(
                    'Default grid for model {} is not found'.format(model_type))
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                os.path.join('..', 'data', 'model_pars.json'))
            with open(file) as f:
                param_grid = json.load(f)[model_name]

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

            if isinstance(current_grid, list) is False:
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
        """Fit model

        Parameters
        ----------
        X: iterable, shape (n_samples, )
            Sequence of tokenized documents

        y: iterable, shape (n_samples, )
            Sequence of labels

        Returns
        -------
        RedditModel
            Fitted model object
        """
        self.classes_ = np.array(sorted(np.unique(y)))
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        """Predict the most likely label

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
        if not self.fitted:
            raise NotFittedError('Model has to be fitted first')
        if not isinstance(X, np.ndarray):
            X = np.array(X, ndmin=1)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict the most likely label

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
        if not self.fitted:
            raise NotFittedError('Model has to be fitted first')
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            indices = X.index
        else:
            indices = list(range(len(X)))

        if not isinstance(X, np.ndarray):
            X = np.array(X, ndmin=1)

        probs = self.model.predict_proba(X)
        probs.index = indices

        return probs

    def get_params(self, deep=None):
        """
        Get parameters of the model

        Returns
        ----------
        dict
            Dictionary with model parameters
        """
        params = {}
        for key in self._get_param_names():
            params[key] = getattr(self, key, None)
        params.update(self.model.get_params())
        return params

    def set_params(self, **params):
        """Set parameters of the model

        Parameters
        ----------
        **params
            Model parameters to update
        """
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.params[key] = value
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
                self.params[key] = value
            else:
                warnings.warn('Parameter {} does not exist'.format(key))
        return self

    def plot_analytics(self, classes=None, fig_sizes=((20, 15), (20, 20)),
                       linkage_pars=None, dendrogram_pars=None,
                       clustering_pars=None, tsne_pars=None,
                       legend_pars=None, label_font_size=17):
        """Plot hieracical clustering dendrogram and T-SNE visualization
        based on the learned class embeddings

        Parameters
        ----------
        classes: iter, optional
            Iterable, contains list of class labels to include to the plots.
            If None, use all classes

        fig_sizes: tuple of tuples, optional
            Figure sizes for plots

        linkage_pars: dict, optional
            Dictionary of parameters for hieracical clustering.
            (scipy.cluster.hierarchy.linkage)

        dendrogram_pars: dict, optional
            Dictionary of parameters for plotting dendrogram.
            (scipy.cluster.hierarchy.dendrogram)

        clustering_pars: dict, optional
            Dictionary of parameters for producing flat clusters.
            (scipy.cluster.hierarchy.fcluster)

        tsne_pars: dict, optional
            Dictionary of parameters for T-SNE.
            (sklearn.manifold.TSNE)

        legend_pars: dict, optional
            Dictionary of parameters for legend plotting
            (matplotlib.pyplot.legend)

        label_font_size: int, optional
            Font size for the labels on T-SNE plot
        """
        if not self.fitted:
            raise NotFittedError('Model has to be fitted first')
        if self.class_embeddings is None:
            raise ValueError(
                'Plotting dendrograms is not available for this class of model')
        if classes is None:
            classes = self.classes_

        if linkage_pars is None:
            linkage_pars = DEFAULT_LINKAGE_PARS
        else:
            linkage_pars = {**DEFAULT_LINKAGE_PARS, **linkage_pars}
        if dendrogram_pars is None:
            dendrogram_pars = DEFAULT_DENDROGRAM_PARS
        else:
            dendrogram_pars = {**DEFAULT_DENDROGRAM_PARS, **dendrogram_pars}
        if clustering_pars is None:
            clustering_pars = DEFAULT_CLUSTERING_PARS
        else:
            clustering_pars = {**DEFAULT_CLUSTERING_PARS, **clustering_pars}
        if tsne_pars is None:
            tsne_pars = DEFAULT_TSNE_PARS
        else:
            tsne_pars = {**DEFAULT_TSNE_PARS, **tsne_pars}
        if legend_pars is None:
            legend_pars = DEFAULT_LEGEND_PARS
        else:
            legend_pars = {**DEFAULT_LEGEND_PARS, **legend_pars}

        z = hac.linkage(self.class_embeddings.loc[classes, :], **linkage_pars)
        fig1 = plt.figure(figsize=fig_sizes[0])
        fancy_dendrogram(z, classes, **dendrogram_pars)

        clusters = fcluster(z, **clustering_pars) - 1
        df_clust = pd.DataFrame({'classes': classes, 'cluster': clusters})
        num_cl = len(df_clust.cluster.unique())
        tsne = TSNE(n_components=2, **tsne_pars)
        Y = tsne.fit_transform(self.class_embeddings.loc[classes, :])
        fig2, ax2 = plt.subplots(figsize=fig_sizes[1])
        colors = cm.jet(np.linspace(0, 1, num_cl))
        for i in range(num_cl):
            ax2.plot(Y[clusters == i, 0], Y[clusters == i, 1],
                     marker='o', linestyle='', color=colors[i])
        ax2.margins(0.05)
        ax2.legend(**legend_pars)
        texts = []
        for i in range(len(classes)):
            texts.append(plt.text(Y[i, 0], Y[i, 1], classes[i],
                                  fontsize=label_font_size))
        adjust_text(texts, arrowprops=dict(
            arrowstyle="-", color='black', lw=0.55))
        return fig1, fig2
