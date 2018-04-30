RedditScore Overview
=========================================

RedditScore is a library that contains tools for building Reddit-based text classification models

RedditScore includes:

    - Document tokenizer with myriads of options, including Reddit- and Twitter-specific options
    - Tools to build and tune the most popular text classification models without any hassle
    - Functions to easily collect Reddit comments from Google BigQuery and Twitter data (including tweets beyond 3200 tweets limit)
    - Instruments to help you build more efficient Reddit-based models and to obtain RedditScores (Nikitin2018_)
    - Tools to use pre-built Reddit-based models to obtain RedditScores for your data

**Note:** RedditScore library and this tutorial are work-in-progress.
`Let me know if you experience any issues <https://github.com/crazyfrogspb/RedditScore/issues>`__.

Usage example:

.. code:: python

    import os

    import pandas as pd

    from redditscore import tokenizer
    from redditscore.models import fasttext_mod

    df = pd.read_csv(os.path.join('redditscore', 'reddit_small_sample.csv'))
    df = df.sample(frac=1.0, random_state=24) # shuffling data
    tokenizer = CrazyTokenizer(hashtags='split') # initializing tokenizer object
    X = df['body'].apply(tokenizer.tokenize) # tokenizing Reddit comments
    y = df['subreddit']

    fasttext_model = fasttext_mod.FastTextModel() # initializing fastText model

    fasttext_model.tune_params(X, y, cv=5, scoring='accuracy') # tune hyperparameters of the model using default grid
    fasttext_model.fit(X, y) # fit model
    fasttext_model.save_model('models/fasttext_model') # save model
    fasttext_model = fasttext.load_model('models/fasttext_model') # load model

    dendrogram_pars = {'leaf_font_size': 14}
    tsne_pars = {'perplexity': 30.0}
    fasttext_model.plot_analytics(dendrogram_pars=dendrogram_pars, # plot dendrogram and T-SNE plot
                             tsne_pars=tsne_pars,
                             fig_sizes=((25, 20), (22, 22)))

    probs = fasttext_model.predict_proba(X)
    av_scores, max_scores = fasttext_model.similarity_scores(X)



References:

.. [Nikitin2018] Nikitin Evgenii, Identyifing Political Trends on Social Media Using Reddit Data, in progress
