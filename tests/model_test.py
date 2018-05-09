import os

import pandas as pd
import pytest

from redditscore.models import fasttext_mod, sklearn_mod
from redditscore.tokenizer import CrazyTokenizer

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..',
                              'redditscore', 'data',
                              'reddit_small_sample.csv'))

tokenizer = CrazyTokenizer(urls='domain')
df['tokens'] = df['body'].apply(tokenizer.tokenize)

pytest.X = df['tokens']
pytest.X_str = df['tokens'].str.join(' ')
pytest.y = df['subreddit']

pytest.MM = sklearn_mod.MultinomialModel(alpha=0.1, random_state=24,
                                         tfidf=False, ngrams=2)
pytest.MM.fit(pytest.X, pytest.y)
pytest.FM = fasttext_mod.FastTextModel(minCount=5)
pytest.FM.fit(pytest.X, pytest.y)
pytest.FM_str = fasttext_mod.FastTextModel(minCount=5)
pytest.FM_str.fit(pytest.X_str, pytest.y)


def test_set_params():
    assert pytest.MM.get_params()['ngrams'] == 2
    pytest.MM.set_params(ngrams=1, alpha=0.5)
    assert pytest.MM.get_params()['ngrams'] == 1
    assert pytest.FM.get_params()['minCount'] == 5
    pytest.FM.set_params(minCount=7, random_state=2018)
    assert pytest.FM.get_params()['minCount'] == 7


def test_multimodel():
    pytest.MM.tune_params(pytest.X, pytest.y, cv=0.2, scoring='neg_log_loss',
                          param_grid={'tfidf': [False, True]})
    pytest.MM.tune_params(pytest.X, pytest.y, cv=5, scoring='accuracy',
                          param_grid={'tfidf': [False, True],
                                      'alpha': [0.1, 1.0]}, refit=True)
    pytest.MM.predict(pytest.X)
    pytest.MM.predict_proba(pytest.X)


def test_fasttext_train():
    pytest.FM.predict(pytest.X)
    pytest.FM.predict_proba(pytest.X)
    pytest.FM.tune_params(pytest.X, pytest.y, cv=0.2, param_grid={
        'step0': {'epoch': [1, 2]},
        'step1': {'minCount': [1, 5]}})


def test_fasttext_str():
    pytest.FM.predict(pytest.X_str)
    pytest.FM.predict_proba(pytest.X_str)


def test_step_exception():
    with pytest.raises(KeyError):
        pytest.FM.tune_params(pytest.X, pytest.y, cv=0.2, param_grid={
            'step0': {'epoch': [1, 2]},
            'step2': {'minCount': [1, 5]}})


def test_dendrogram():
    pytest.FM.plot_analytics()


def test_save_load():
    pytest.MM.save_model('multi.pkl')
    pytest.FM.save_model('fasttext')
    pytest.MM = sklearn_mod.load_model('multi.pkl')
    pytest.FM = fasttext_mod.load_model('fasttext')
    os.remove('multi.pkl')
    os.remove('fasttext.pkl')
    os.remove('fasttext.bin')
