# RedditScore
Package for performing Reddit-based text analysis

Includes:
- Document tokenizer with myriads of options, including Reddit- and Twitter-specific options
- Tools to build and tune the most popular text classification models without any hassle
- Functions to easily collect Reddit comments from Google BigQuery and Twitter data (including tweets beyond 3200 tweets limit)
- Instruments to help you build more efficient Reddit-based models and to obtain RedditScores
- Tools to use pre-built Reddit-based models to obtain RedditScores for your data

Full documentation and tutorials live here: http://redditscore.readthedocs.io

To install package:

	pip install git+https://github.com/crazyfrogspb/RedditScore.git

If you want to be able all features of the library, also install these
dependencies:

	pip install Cython pybind11 selenium keras tensorflow tensorflow-gpu nltk pandas-gbq
	pip install git+https://github.com/crazyfrogspb/tweepy.git
	git clone git+https://github.com/crazyfrogspb/fastText.git
	cd fastText
	pip install .

- Cython, pybind11, fasttext - for training fastText models
- keras, tensorflow, tensorflow-gpu - for training neural networks
- nltk - for using stemming and NLTK stopwords lists
- pansas-gbq - for collecting Reddit data
- selenium, tweepy - for collecting Twitter data

To cite:

    {
      @misc{Nikitin2018,
      author = {Nikitin, E.},
      title = {RedditScore},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/crazyfrogspb/RedditScore}}
    }
