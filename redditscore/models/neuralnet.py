import copy

import numpy as np
from keras.initializers import RandomNormal
from keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from . import redditmodel, sklearn


def reverse_dictionary(dict):
    return {v: k for k, v in dict.items()}


class MLPModel(redditmodel.RedditModel):
    def __init__(self, n_layers=0, layer_sizes=32, dim=100, embeddings=None,
                 maxlen=50, activation='linear', dropout=False, ngrams=1,
                 min_count=1, lr=0.001, random_state=24, verbose=False):
        super().__init__(random_state=random_state)
        if not isinstance(n_layers, int):
            raise ValueError("n_layers must be an integer")
        if not isinstance(dim, int):
            raise ValueError("dim must be an integer")

        if isinstance(layer_sizes, int):
            self.layer_sizes = [layer_sizes] * n_layers
        elif isinstance(layer_sizes, list):
            if len(layer_sizes) != n_layers:
                raise ValueError(
                    'Length of layer_sizes must be equal to n_layers')
            else:
                self.layer_sizes = layer_sizes
        else:
            raise ValueError("layer_sizes must be an integer or a list")

        self.n_layers = n_layers
        self.dim = dim
        self.embeddings = embeddings
        self.activation = activation
        self.dropout = dropout
        self.maxlen = maxlen
        self.ngrams = ngrams
        self.min_count = min_count
        self.lr = lr
        self.verbose = verbose

        self._vocab_size = 0
        self._num_classes = None
        self._model = None
        self._analyzer = None
        self._word2idx = {}
        self._idx2word = {}
        self._encoder = LabelEncoder()

    def _build_vocab(self, X):
        vectorizer = CountVectorizer(
            analyzer=sklearn._build_analyzer(self.ngrams),
            min_df=self.min_count)
        self._analyzer = vectorizer.build_analyzer()
        vectorizer.fit(X)
        for key, value in vectorizer.vocabulary_.items():
            self._word2idx[key] = value
        self._idx2word = reverse_dictionary(self._word2idx)
        self._vocab_size = len(self._word2idx)

    def _tokens_to_inds(self, X):
        X_train = []
        for doc in X:
            doc_ind = []
            for x in self._analyzer(doc):
                ind = self._word2idx.get(x)
                if ind:
                    doc_ind.append(ind)
            X_train.append(doc_ind)

        X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        return X_train

    def _build_model(self):
        model = Sequential()

        if self.embeddings is None:
            model.add(Embedding(self._vocab_size, self.dim,
                                embeddings_initializer=RandomNormal(seed=self.random_state)))
        else:
            model.add(Embedding(self._vocab_size, self.dim,
                                weights=[self.weights], trainable=False))

        model.add(GlobalAveragePooling1D())
        if self.n_layers > 0:
            model.add(Dense(self.layer_sizes[0], input_shape=(dim, )))
            for i in range(1, self.n_layers):
                model.add(Dense(self.layer_sizes[i]))
                if self.dropout:
                    model.add.Dropout(self.dropout)
        model.add(Dense(self._num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(self.lr),
                      metrics=['accuracy'])
        self._model = model

    def fit(self, X, y, batch_size=32, epochs=5, validation_data=None):

        self._num_classes = len(np.unique(y))
        if self.verbose:
            print('Building vocabularly')
        self._build_vocab(X)
        X_train = np.asarray(self._tokens_to_inds(X))
        y_train = self._encoder.fit_transform(y)
        y_train = to_categorical(y_train)
        if self.verbose:
            print('Compiling model')
        self._build_model()

        if validation_data is not None:
            X_val = np.asarray(self._tokens_to_inds(validation_data[0]))
            y_val = self._encoder.transform(validation_data[1])
            y_val = to_categorical(y_val)
            self._model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_val, y_val),
                            verbose=self.verbose)
        else:
            self._model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=self.verbose)


class CNNModel(redditmodel.RedditModel):
    pass


class LSTMModel(redditmodel.RedditModel):
    pass
