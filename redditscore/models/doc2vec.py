    from random import shuffle

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression

from . import redditmodel


class Doc2VecModel(redditmodel.RedditModel):
    def __init__(self, random_state=24, dm=0, vector_size=100, window=5,
                 negative=5, hs=0, min_count=5, sample=1e-5, epochs=10,
                 dbow_words=0, workers=8, steps=1000, alpha=0.025):
        super().__init__(random_state=random_state)
        self.model_type = 'doc2vec'
        self._model = LogisticRegression()

        self.dm = dm
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.hs = hs
        self.min_count = min_count
        self.sample = sample
        self.epochs = epochs
        self.dbow_words = dbow_words
        self.workers = workers
        self.steps = steps
        self.alpha = alpha

    @staticmethod
    def _prepare_data(X, y):
        alldocs = []
        total_documents = 0
        for doc, label in zip(X, y):
            td = TaggedDocument(doc, [label])
            alldocs.append(td)
            total_documents += 1
        return alldocs, total_documents

    def _calc_sims(self, docs, steps, alpha):
        vectors = np.zeros((len(docs), self.vector_size))
        for i, doc in enumerate(docs):
            vectors[i, :] = self._doc2vec.infer_vector(doc, steps=steps,
                                                       alpha=alpha)
        sims = 1 - cdist(vectors, self.class_embeddings, metric='cosine')
        return pd.DataFrame(sims, columns=self._classes)

    def fit(self, X, y):
        self._doc2vec = Doc2Vec(dm=self.dm, vector_size=self.vector_size,
                                window=self.window, negative=self.negative,
                                hs=self.hs, min_count=self.min_count,
                                sample=self.sample, epochs=self.epochs,
                                dbow_words=self.dbow_words,
                                workers=self.workers)

        self._classes = sorted(np.unique(y))
        alldocs, total_documents = self._prepare_data(X, y)
        doclist = alldocs[:]

        self._doc2vec.build_vocab(alldocs)
        np.random.seed(self.random_state)
        shuffle(doclist)
        self._doc2vec.train(doclist, epochs=self._doc2vec.epochs,
                            total_examples=total_documents)

        emb = np.zeros((len(self._classes), self.vector_size))
        for i, class_label in enumerate(self._classes):
            emb[i, :] = self._doc2vec.docvecs[class_label]
        self.class_embeddings = pd.DataFrame(emb, index=self._classes)

        sims = self._calc_sims(X, self.steps, self.alpha)
        self._model.fit(sims, y)
        self.fitted = True

    def predict(self, X):
        sims = self._calc_sims(X, self.steps, self.alpha)
        return self._model.predict(sims)

    def predict_proba(self, X):
        sims = self._calc_sims(X, self.steps, self.alpha)
        return self._model.predict_proba(sims)
