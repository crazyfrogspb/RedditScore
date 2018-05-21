# -*- coding: utf-8 -*-
"""
bow_mod: A wrapper for Bag-of-Words models

Author: Evgenii Nikitin <e.nikitin@nyu.edu>

Part of https://github.com/crazyfrogspb/RedditScore project

Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""

import dill as pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from . import redditmodel


def load_model(filepath):
    """Loan pickled instance of SklearnModel.

    Parameters
    ----------
    filepath : str
        Path to the pickled model file.

    Returns
    -------
    SklearnModel
        Unpickled model.

    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


class BoWModel(redditmodel.RedditModel):
    """A wrapper for Bag-of-Words models with or without tf-idf re-weighting
    Parameters
    ----------
    estimator: scikit-learn model
        Estimator object (classifier or regressor)

    ngrams: int, optional
        The upper boundary of the range of n-values for different n-grams to be extracted

    tfidf: bool, optional
        If true, use tf-idf re-weighting

    random_state: integer, optional
        Random seed

    **kwargs
         Parameters of the multinomial model. For details check scikit-learn documentation.
    Attributes
    ----------
    params : dict
        Dictionary with model parameters
    """

    def __init__(self, estimator, ngrams=1, tfidf=True, random_state=24):
        super().__init__(random_state=random_state)
        self.ngrams = ngrams
        self.tfidf = tfidf

        if self.tfidf:
            vectorizer = TfidfVectorizer(
                analyzer=self._build_analyzer(self.ngrams))
        else:
            vectorizer = CountVectorizer(
                analyzer=self._build_analyzer(self.ngrams))
        self.model = Pipeline(
            [('vectorizer', vectorizer), ('model', estimator)])

    def set_params(self, **params):
        """
        Set the parameters of the model.

        Parameters
        ----------
        **params: {'tfidf', 'ngrams', 'random_state'} or
            parameters of the corresponding models
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.model.named_steps['model'], key):
                setattr(self.model.named_steps['model'], key, value)

        if self.tfidf:
            vectorizer = TfidfVectorizer(
                analyzer=self._build_analyzer(self.ngrams))
        else:
            vectorizer = CountVectorizer(
                analyzer=self._build_analyzer(self.ngrams))

        self.model = Pipeline(
            [('vectorizer', vectorizer),
             ('model', self.model.named_steps['model'])])

        return self

    def save_model(self, filepath):
        """Save model to disk.

        Parameters
        ----------
        filepath : str
            Path to the file where the model will be sabed.

        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def _build_analyzer(ngrams):
        # Build analyzer for vectorizers for a given ngram range
        return lambda doc: redditmodel.word_ngrams(doc, (1, ngrams))
