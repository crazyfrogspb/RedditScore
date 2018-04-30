Installation
=========================================

To install the package, run the following command:

>>> pip install git+https://github.com/crazyfrogspb/RedditScore.git

If you want to use all features of the RedditScore library, make sure to install these extra dependencies:

    - For training fastText models:
        >>> pip install Cythin pybind11
        >>> git clone git+https://github.com/crazyfrogspb/fastText.git
        >>> cd fastText
        >>> pip install .
    - For training neural networks:
        >>> pip install tensorflow tensorflow-gpu keras
    - For collecting Reddit data from BigQuery:
        >>> pip install pandas-gbq
    - For collecting Twitter data:
        >>> pip install selenium
        >>> pip install git+https://github.com/crazyfrogspb/tweepy.git
    - For using stemming or NLTK stopwords lists:
        >>> pip install nltk