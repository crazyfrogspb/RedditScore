CrazyTokenizer
=====================

Tokenizer description
--------------------

CrazyTokenizer is a part of `RedditScore project <https://github.com/crazyfrogspb/RedditScore>`__.
It's a tokenizer - tool for splitting strings of text into tokens. Tokens can
then be used as input for a variety of machine learning models.
CrazyTokenizer was developed specifically for tokenizing Reddit comments and
tweets, and it includes many features to deal with these types of documents.
Of course, feel free to use for any other kind of text data as well.

Initializing
--------------------
To import and to initialize an instance of CrazyTokenizer with the default
preprocessing options, do the following:

.. code:: python

    from redditscore.tokenizer import CrazyTokenizer
    tokenizer = CrazyTokenizer()

Now you can start tokenizing!

.. code:: python

    tokenizer.tokenizer("(@crazyfrogspb Hey,dude, have you heard that
    https://github.com/crazyfrogspb/RedditScore
    is the best Python library ever??)"

.. parsed-literal::

    ['TOKENTWITTERHANDLE', 'hey', 'dude', 'have', 'you', 'heard', 'that',
    'github_domain', 'is', 'the', 'best', 'python', 'library', 'ever']
