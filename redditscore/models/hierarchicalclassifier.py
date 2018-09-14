import os
import pickle
from abc import ABCMeta
from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import to_tree
from tqdm import tqdm

import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from . import fasttext_mod

DEFAULT_LINKAGE_PARS = {'method': 'average',
                        'metric': 'cosine', 'optimal_ordering': True}


class HierarchicalClassifier(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    def __init__(self, class_embeddings, class_labels, estimator,
                 linkage_pars=None, random_state=24, verbose=True,
                 models_dir=None, downsample=False):
        self.estimator = estimator
        if self.estimator.__class__.__name__ == 'FastTextModel':
            self.fasttext_ = True
        else:
            self.fasttext_ = False
        self.fitted = False
        self.random_state = random_state
        self.verbose = verbose
        self.models_dir = models_dir
        self.downsample = downsample

        self.labels_dict = {}
        for i, class_label in enumerate(class_labels):
            self.labels_dict[i] = class_label

        if linkage_pars is None:
            linkage_pars = DEFAULT_LINKAGE_PARS
        else:
            linkage_pars = {**DEFAULT_LINKAGE_PARS, **linkage_pars}
        self.z = hac.linkage(class_embeddings, **linkage_pars)

        self._build_graph()

    def _build_graph(self):
        self.root_ = to_tree(self.z)
        self.root_id_ = str(self.root_.id)
        self.graph_ = nx.DiGraph()
        self.graph_.add_node(self.root_id_)
        for node in self._walk(self.root_):
            label = str(self.labels_dict.get(node.id, node.id))
            self.graph_.nodes[label]['model'] = None
            self.graph_.nodes[label]['flat_classes'] = list(
                map(self.labels_dict.get, node.pre_order()))
            self.graph_.nodes[label]['left'] = None
            self.graph_.nodes[label]['right'] = None
            if node.left:
                label_left = str(self.labels_dict.get(
                    node.left.id, node.left.id))
                self.graph_.add_node(label_left)
                self.graph_.add_edge(label, label_left)
                self.graph_.nodes[label]['left'] = label_left
            if node.right:
                label_right = str(self.labels_dict.get(
                    node.right.id, node.right.id))
                self.graph_.add_node(label_right)
                self.graph_.add_edge(label, label_right)
                self.graph_.nodes[label]['right'] = label_right

        self.classes_ = list(
            node
            for node in self.graph_.nodes()
            if node != self.root_id_
        )

        self.paths_ = {}
        for class_ in self.classes_:
            self.paths_[class_] = nx.shortest_path(
                self.graph_, self.root_id_, class_)

    @staticmethod
    def _walk(node):
        queue = deque([node])
        while queue:
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            yield node

    @property
    def n_classes_(self):
        return len(self.classes_)

    def _train_estimator(self, node, X, y):
        X_ = X.copy()
        y_ = pd.Series().reindex_like(y)
        for succ in self.graph_.successors(node):
            y_.loc[y.isin(self.graph_.nodes[succ]['flat_classes'])] = succ

        y_.dropna(inplace=True)
        X_ = X_.loc[y_.index]

        if self.downsample:
            sample_size = y_.value_counts().min()
            X_ = X_.groupby(y_).apply(lambda x: x.sample(
                sample_size, random_state=self.random_state)).reset_index(level=0, drop=True)
            y_ = y_.loc[X_.index]

        clf_ft = deepcopy(self.estimator)
        clf_ft.fit(X_, y_)
        return clf_ft

    def fit(self, X, y):
        X_ = X.reset_index(drop=True)
        y_ = y.reset_index(drop=True)
        for node in self.graph_.nodes:
            if self.graph_.out_degree(node) != 0:
                if self.verbose:
                    print('Fitting model for class {}'.format(node))
                if self.models_dir:
                    clf = self._train_estimator(node, X_, y_)
                    model_path = os.path.join(
                        self.models_dir, 'model_{}'.format(node))
                    if self.fasttext_:
                        clf.save_model(model_path)
                    else:
                        with open(model_path + '.pkl', 'wb') as f:
                            pickle.dump(clf, f)
                    self.graph_.nodes[node]['model'] = model_path
                else:
                    self.graph_.nodes[node]['model'] = self._train_estimator(
                        node, X_, y_)
            else:
                if self.verbose:
                    print('Reached terminal node for class {}'.format(node))

        self.fitted = True

    def load_model(self, node):
        model_path = os.path.join(self.models_dir, 'model_{}'.format(node))
        if not os.path.exists(model_path + '.pkl'):
            return None
        if self.fasttext_:
            clf = fasttext_mod.load_model(model_path)
        else:
            with open(model_path + '.pkl', 'rb') as f:
                clf = pickle.load(f)
        return clf

    def _generate_prob_matrix(self, X):
        split_nodes = [
            node for node in self.graph_.nodes if self.graph_.out_degree(node) > 0]
        predictions = pd.DataFrame(
            index=X.index, columns=split_nodes)
        for node in split_nodes:
            if self.models_dir:
                clf = self.load_model(node)
            else:
                clf = self.graph_.nodes[node]['model']
            probs = clf.predict_proba(X)
            mapping = {self.graph_.nodes[node]['left']: 'left',
                       self.graph_.nodes[node]['right']: 'right'}
            probs.rename(columns=mapping, inplace=True)
            predictions.loc[:, node] = probs['left'].copy()

        return predictions

    def predict(self, X):
        if not self.fitted:
            raise NotFittedError('Model has to be fitted first')

        predictions = self.predict_proba(X)

        return predictions.idxmax(axis=1)

    def predict_proba(self, X):
        if not self.fitted:
            raise NotFittedError('Model has to be fitted first')

        predictions = pd.DataFrame(
            index=X.index, columns=self.labels_dict.values())
        prob_matrix = self._generate_prob_matrix(X)
        for i, class_label in self.labels_dict.items():
            probabilities = np.ones((len(X), ))
            prev_node = self.paths_[class_label][0]
            for cur_node in self.paths_[class_label][1:]:
                if self.graph_.nodes[prev_node]['left'] == cur_node:
                    probabilities *= prob_matrix.loc[:, prev_node]
                else:
                    probabilities *= (1 - prob_matrix.loc[:, prev_node])
                prev_node = cur_node
            predictions.loc[:, class_label] = probabilities

        return predictions
