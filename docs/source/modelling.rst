Modelling
==========

You collected your training data, coverted it to the lists of tokens, now you can start training models!

RedditScore currently supports training the following types of models:

    - Any models from `scikit-learn package <http://scikit-learn.org>`__
    - `fastText model <https://github.com/facebookresearch/fastText>`__
    - MLP, LSTM, CNN neural networks (Keras implementation) - TODO

Fitting models
---------------------

All model wrappers have very similar interface:

.. code:: python

    from redditscore import tokenizer
    from redditscore.models import fasttext_mod, sklearn_mod
    from sklearn.naive_bayes import MultinomialNB

    # reading and tokenizing data
    df = pd.read_csv(os.path.join('redditscore', 'reddit_small_sample.csv'))
    df = df.sample(frac=1.0, random_state=24) # shuffling data
    tokenizer = CrazyTokenizer(splithashtags=True) # initializing tokenizer object
    X = df['body'].apply(tokenizer.tokenize) # tokenizing Reddit comments
    y = df['subreddit']

    fasttext_model = fasttext_mod.FastTextModel(epochs=5, minCount=5)
    multi_model = sklearn_mod.SklearnModel(estimator=MultinomialNB(alpha=0.1), ngrams=2, tfidf=False)

    fasttext_model.fit(X, y)
    multi_model.fit(X, y)

Model persistence
---------------------
To save the model, use ``save_model`` method:

>>> fasttext_model.save_model('models/fasttext')
>>> multi_model.save_model('models/multi.pkl')

Each module has its own ``load_model`` function:

>>> fasttext_model = fasttext_mod.load_model('models/fasttext')
>>> multi_model = sklearn_mod.load_model('models/multi.pkl')

**Note**: fastText and Keras models are saved into two files with '.pkl' and '.bin' extensions.read

Predictions and similarity scores
---------------------------------
Using models for prediction is very straightforward:

>>> pred_labels = fasttext_model.predict(X)
>>> pred_probs = fasttext_model.predict_proba(X)

Both ``predict`` and ``predict_proba`` return pandas DataFrames with class labels as column names.

Model tuning and validation
---------------------------
Each model class has ``cv_score`` and ``tune_params`` methods. You can use these methods to assess the quality of your model
and to perform tuning of hyperparameters.

``cv_score`` method has two optional arguments - ``cv`` and ``scoring``. ``cv`` argument can be:
    - float: score will be calculated using a randomly sampled holdout set containing corresponding proportion of ``X``
    - None: use 3-fold cross-validation
    - integer: use n-fold cross-validation
    - an object to be used as a cross-validation generator: for example, KFold from sklearn
    - an iterable yielding train and test split: for example, list of tuples with train and test indices

``scoring`` can be either `a string <http://scikit-learn.org/stable/modules/model_evaluation.html>`__ or a scorer callable object.

Examples:

>>> fasttext_model.cv_score(X, y, cv=0.2, scoring='accuracy')
>>> fasttext_model.cv_score(X, y, cv=5, scoring='neg_log_loss')

``tune_params`` can help you to perform grid search over the grid of hyperparameters. In addition to ``cv`` and ``scoring`` parameters it also
has three additional optional parameters: ``param_grid``, ``verbose``, and ``refit``.

``param_grid`` can have a few different structures:
    - None: if None, use default step grid for the corresponding model
    - dictionary with parameters names as keys and lists of parameter settings as values.
    - dictionary with enumerated steps of grid search. Each step has to be a dictionary with parameters names as keys and lists of parameter settings as values.

Examples:

>>> fasttext_model.tune_params(X, y)
>>> param_grid = {'epochs': [1,5,10], 'dim': [50,100,300]}
>>> fasttext_model.tune_params(X, y, param_grid=param_grid)
>>> param_grid = {'step0': param_grid, 'step1': {'t': [1e-4, 1e-3, 1e-3]}}

If ``verbose`` is True, messages with grid process results will be printed.
If ``refit`` is True, the model will be refit with the full data after grid search is over.

``tune_params`` returns a tuple: dictionary with the best found parameters and the best value of the chosen metric.
In addition, original model parameters will be replaced inplace with the best found parameters.

Visualization of the class embeddings
-------------------------------------

For fastText and neural network models, you can visualize resulting class embeddings. This might be useful for a couple of reasons:
    - It can be used as an informal way to confirm that the model was able to learn meaningful semantic differences between classes. In particular, classes that one expects to be more semantically similar should have similar class embeddings.
    - It can help you to group different classes together. This is particularly useful for building Reddit-based models and calculating RedditScores. There are a lot of different subreddits, but a lot of them are quite similar to each other (say, /r/Conservaitve and /r/republicans). Visualizations can help you to identify similar subreddits, which can be grouped together for improved predictive performance.

.. figure:: figures/dendrogram.png
   :alt: Dengrogram for class embeddings

   Dengrogram for class embeddings

.. figure:: figures/dendrogram.png
   :alt: t-SNE visualization

   t-SNE visualization
