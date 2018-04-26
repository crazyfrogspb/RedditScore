# -*- coding: utf-8 -*-
"""
SklearnModel: A wrapper for several text classification models from Scikit-Learn

Author: Evgenii Nikitin <e.nikitin@nyu.edu>

Part of https://github.com/crazyfrogspb/RedditScore project

Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""

import dill as pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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


class SklearnModel(redditmodel.RedditModel):
    """SVM and Naive Bayes classifier for multinomial and Bernoulli models
    with or without tf-idf re-weighting

    Parameters
    ----------
    model_type: {'multinomial', 'bernoulli', 'svm'}, optional
        Model type

        - 'multinomial': MultinomialNB
        - 'bernoulli': BernoulliNB
        - 'svm': SVC

    ngrams: int, optional
        The upper boundary of the range of n-values for different n-grams to be extracted

    tfidf: bool, optional
        If true, use tf-idf re-weighting

    random_state: integer, optional
        Random seed

    **kwargs
         Parameters of the corresponding models. For details check scikit-learn documentation.

    Attributes
    ----------
    params : dict
        Dictionary with model parameters
    """

    def __init__(self, model_type='multinomial', ngrams=1, tfidf=True, random_state=24, **kwargs):
        super().__init__(random_state=random_state)
        self.params = {}
        self.ngrams = ngrams
        self.tfidf = tfidf
        if model_type not in ['multinomial', 'bernoulli', 'svm']:
            raise ValueError('{} is not supported yet'.format(model_type))
        self.model_type = model_type
        self.set_params(**kwargs)

    def set_params(self, **params):
        """
        Set the parameters of the model.

        Parameters
        ----------
        **params: {'tfidf', 'ngrams', 'random_state'} or
            parameters of the corresponding models
        """
        for key in ['tfidf', 'ngrams', 'random_state']:
            if key in params:
                setattr(self, key, params[key])
                params.pop(key)
        self.params.update(params)

        if self.model_type == 'multinomial':
            model = MultinomialNB(**self.params)
        elif self.model_type == 'bernoulli':
            model = BernoulliNB(**self.params)
        elif self.model_type == 'svm':
            model = SVC(**self.params)

        if self.tfidf:
            vectorizer = TfidfVectorizer(
                analyzer=self._build_analyzer(self.ngrams))
        else:
            vectorizer = CountVectorizer(
                analyzer=self._build_analyzer(self.ngrams))

        self._model = Pipeline([('vectorizer', vectorizer), ('model', model)])

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
