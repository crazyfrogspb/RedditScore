RedditScore Overview
=========================================

RedditScore is a library that contains tools for building Reddit-based text classification models

RedditScore includes:
    - Document tokenizer with myriads of options, including Reddit- and Twitter-specific options
    - Tools to build and tune most popular text classification models without any hassle
    - Instruments to help you build more efficient Reddit-based models and to obtain RedditScores (Nikitin2018_)

**Note:** RedditScore library and this tutorial are work-in-progress.
`Let me know if you experience any issues <https://github.com/crazyfrogspb/RedditScore/issues>`__.

Usage example:

.. code:: python

    import os

    import pandas as pd

    from redditscore import tokenizer, models

    df = pd.read_csv(os.path.join('redditscore', 'reddit_small_sample.csv'))
    tokenizer = CrazyTokenizer(urls='domain', splithashtags=True)
    df['tokens'] = df['body'].apply(tokenizer.tokenize)
    X = df['tokens']
    y = df['subreddit']

    multi_model = sklearn.SklearnModel(
        model_type='multinomial', alpha=0.1, random_state=24, tfidf=False, ngram_range=(1, 1))
    fasttext_model = fasttext.FastTextModel(minCount=5)

    multi_model.tune_params(X, y, cv=5, scoring='neg_log_loss')
    fasttext_model.fit(X, y)


References:

.. [Nikitin2018] Nikitin Evgenii, Identyifing Political Trends on Social Media Using Reddit Data, in progress
