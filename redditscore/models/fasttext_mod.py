# -*- coding: utf-8 -*-
"""
FastTextModel: A wrapper for Facebook fastText model

Author: Evgenii Nikitin <e.nikitin@nyu.edu>

Part of https://github.com/crazyfrogspb/RedditScore project

Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""

import os
import pickle
import tempfile
import warnings

import fastText
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import QuantileTransformer

from . import redditmodel


def chunking_dot(big_matrix, small_matrix, chunk_size=50000):
    # dot product in chunks
    small_matrix = np.ascontiguousarray(small_matrix)
    R = np.empty((big_matrix.shape[0], small_matrix.shape[1]))
    for i in range(0, R.shape[0], chunk_size):
        end = i + chunk_size
        R[i:end] = np.dot(big_matrix[i:end], small_matrix)
    return R


def load_model(filepath):
    """Load pickled model.

    Parameters
    ----------
    filepath : str
        Path to the file where the model will be saved. NOTE: the
        directory has to contain two files with provided name:
        with '.pkl' and 'bin' file extensions.

    Returns
    -------
    FastTextModel
        Unpickled model object.

    """
    with open(os.path.splitext(filepath)[0] + '.pkl', 'rb') as f:
        model = pickle.load(f)
    model._model._model = fastText.load_model(
        os.path.splitext(filepath)[0] + '.bin')
    return model


def _data_to_temp(X, label, y=None):
    # Generate temorary file
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as tmp:
        for i in range(X.shape[0]):
            if y is not None:
                doc = label + y[i] + ' '
                if isinstance(X[i], list):
                    doc += ' '.join(X[i])
                elif isinstance(X[i], str):
                    doc += X[i]
                else:
                    raise ValueError(
                        'X has to be a sequence of tokens or strings')
            else:
                if isinstance(X[i], list):
                    doc = ' '.join(X[i])
                elif isinstance(X[i], str):
                    doc = X[i]
                else:
                    raise ValueError(
                        'X has to be a sequence of tokens or strings')
            tmp.write("%s\n" % doc)
    return path


class FastTextClassifier(BaseEstimator, ClassifierMixin):
    # Auxiliary sklearn-style wrapper for fastText python library
    def __init__(self, lr=0.1,
                 dim=100,
                 ws=5,
                 epoch=5,
                 minCount=1,
                 minCountLabel=0,
                 minn=0,
                 maxn=0,
                 neg=5,
                 wordNgrams=1,
                 loss="softmax",
                 bucket=2000000,
                 thread=12,
                 lrUpdateRate=100,
                 t=1e-4,
                 label="__label__",
                 verbose=2):
        self.lr = lr
        self.dim = dim
        self.ws = ws
        self.epoch = epoch
        self.minCount = minCount
        self.minCountLabel = minCountLabel
        self.minn = minn
        self.maxn = maxn
        self.neg = neg
        self.wordNgrams = wordNgrams
        self.loss = loss
        self.bucket = bucket
        self.thread = thread
        self.lrUpdateRate = lrUpdateRate
        self.t = t
        self.label = label
        self.verbose = verbose

        self._model = None
        self._num_classes = None

    def fit(self, X, y):
        # Fit model
        path = _data_to_temp(X, self.label, y)
        self._num_classes = len(np.unique(y))
        self._model = fastText.train_supervised(path,
                                                lr=self.lr,
                                                dim=self.dim,
                                                ws=self.ws,
                                                epoch=self.epoch,
                                                minCount=self.minCount,
                                                minCountLabel=self.minCountLabel,
                                                minn=self.minn,
                                                maxn=self.maxn,
                                                neg=self.neg,
                                                wordNgrams=self.wordNgrams,
                                                loss=self.loss,
                                                bucket=self.bucket,
                                                thread=self.thread,
                                                lrUpdateRate=self.lrUpdateRate,
                                                t=self.t,
                                                label=self.label,
                                                verbose=self.verbose)
        os.remove(path)
        return self

    def predict(self, X):
        # Return predictions
        if isinstance(X[0], list):
            docs = [' '.join(doc) for doc in X]
        elif isinstance(X[0], str):
            docs = list(X)
        else:
            raise ValueError("X has to contrain sequence of tokens or strings")
        predictions = self._model.predict(docs, k=1)[0]
        return np.array([pred[0][len(self.label):]
                         for pred in predictions])

    def predict_proba(self, X):
        # Return predicted probabilities
        if isinstance(X[0], list):
            docs = [' '.join(doc) for doc in X]
        elif isinstance(X[0], str):
            docs = list(X)
        else:
            raise ValueError("X has to contrain sequence of tokens or strings")
        predictions = zip(*self._model.predict(docs, k=self._num_classes))
        probabilities = []
        for pred in predictions:
            d = {key[len(self.label):]: value for key, value in zip(*pred)}
            probabilities.append(d)
        return pd.DataFrame(probabilities).fillna(1e-10)


class FastTextModel(redditmodel.RedditModel):
    """Facebook fastText classifier

    Parameters
    ----------
    random_state : int, optional
        Random seed (the default is 24).
    **kwargs
        Other parameters for fastText model.
        Full description can be found here:
        https://github.com/facebookresearch/fastText
    """

    def __init__(self, random_state=24, **kwargs):
        super().__init__(random_state=random_state)
        self.model_type = 'fasttext'
        self._score_scaler = None
        self._model = FastTextClassifier(**kwargs)

    def set_params(self, **params):
        """Set parameters of the model

        Parameters
        ----------
        **params
            Model parameters to update
        """
        self._model.set_params(**params)

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
        FastTextModel
            Fitted model object
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        self._classes = np.array(sorted(np.unique(y)))
        self._model.fit(X, y)
        fd, path = tempfile.mkstemp()
        self._model._model.save_softmax(path)
        emb = pd.read_csv(
            path, skiprows=[0], delimiter=' ', header=None).dropna(axis=1)
        emb = emb.round(decimals=5)
        emb[0] = emb[0].str[len(self._model.label):]
        emb.set_index(0, inplace=True)
        self.class_embeddings = emb.loc[self._classes]
        os.remove(path)
        self.fitted = True
        return self

    def save_model(self, filepath):
        """Save model to disk.

        Parameters
        ----------
        filepath : str
            Path to the file where the model will be saved. NOTE:
            The model will be saved in two files: with '.pkl' and 'bin'
            file extensions.

        """
        with open(os.path.splitext(filepath)[0] + '.pkl', 'wb') as f:
            pickle.dump(self, f)
        self._model._model.save_model(os.path.splitext(filepath)[0] + '.bin')
