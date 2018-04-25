Modelling
==========

You collected your training data, coverted it to the lists of tokens, now you can start training models!

RedditScore currently supports training the following types of models:

    - Multinomial Naive Bayes, Bernoulli Naive Bayes and SVM from `scikit-learn package <http://scikit-learn.org>`__
    - `fastText model <https://github.com/facebookresearch/fastText>`__
    - MLP, LSTM, CNN neural networks (Keras implementation) - TODO

Fitting models
---------------------

All model wrappers have very similar interface:

.. code:: python

    from redditscore import tokenizer
    from redditscore.models import fasttext_mod, sklearn_mod

    # reading and tokenizing data
    df = pd.read_csv(os.path.join('redditscore', 'reddit_small_sample.csv'))
    df = df.sample(frac=1.0, random_state=24) # shuffling data
    tokenizer = CrazyTokenizer(splithashtags=True) # initializing tokenizer object
    X = df['body'].apply(tokenizer.tokenize) # tokenizing Reddit comments
    y = df['subreddit']

    fasttext_model = fasttext_mod.FastTextModel(epochs=5, minCount=5)
    multi_model = sklearn_mod.SklearnModel(model_type='multinomial', ngrams=2, tfidf=False)

    fasttext_model.fit(X, y)
    multi_model.fit(X, y)

Model persistence
---------------------
To save the model:

>>> fasttext_model.save_model('models/fasttext')
>>> multi_model.save_model('models/multi.pkl')

Each module has its own ``load_model`` function:

>>> fasttext_model = fasttext_mod.load_model('models/fasttext')
>>> multi_model = sklearn_mod.load_model('models/multi.pkl')

**Note**: fastText and Keras models are saved into two files with '.pkl' and '.bin' extensions.read
