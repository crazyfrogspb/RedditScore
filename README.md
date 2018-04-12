# RedditScore
Package for performing Reddit-based text analysis

Includes:
- Document tokenizer with myriads of options, including Reddit- and Twitter-specific options
- Tools to build and tune most popular text classification models without any hassle
- Instruments to help you build more efficient Reddit-based models and to obtain RedditScores
- Tools to use pre-built Reddit-based models to obtain RedditScores for your data

Full documentation lives here: http://redditscore.readthedocs.io

Tokenizer usage:

	from redditscore.tokenizer import CrazyTokenizer
	trump_rant = "@realDonaldTrump #fucktrump WHO ELECTED this Guy?! 😭'"
	tokenizer = CrazyTokenizer(ignorestopwords='english', splithashtags=True, neg_emojis=True, hashtags=False)
	tokenizer.tokenize_doc(trump_rant)

	Output:
	['TOKENTWITTERHANDLE', 'fuck', 'trump', 'WHO', 'ELECTED', 'guy', 'NEG_EMOJI']


Model usage:

    import pandas as pd
    import os
    from redditscore.tokenizer import CrazyTokenizer
    import pandas as pd
    from redditscore.model import BayesModel

    df = pd.read_csv(os.path.join('redditscore', 'data', 'reddit_small_sample.csv'))
    tokenizer = CrazyTokenizer(urls='domain')
    df['tokens'] = df['body'].apply(tokenizer.tokenize)

    model = BayesModel(multi_model=True, alpha=1.0e-10, random_state=24, tfidf=True, ngram_range=(1,1))
    X = df['tokens']
    y = df['subgroup']
    model.tune_params(X, y, cv=5)


To install package:

	pip install git+https://github.com/crazyfrogspb/RedditScore.git

To perform complete installation with all features:

  pip install git+https://github.com/crazyfrogspb/RedditScore.git#egg=redditscore[nltk,neural_nets,fasttext]

To cite:

    {
      @misc{Nikitin2018,
      author = {Nikitin, E.},
      title = {RedditScore},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/crazyfrogspb/RedditScore}},
      commit = {faf15eaed7cb334dfd7213195bbbb68861767d6a}
    }
