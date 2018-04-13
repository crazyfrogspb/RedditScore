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

CrazyTokenizer is based on the amazing `spaCY NLP framework <https://spacy.io/`__.
Make sure to check it out!

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

Features
--------------------

Lowercasing and all caps
^^^^^^^^
For many text classification problems, keeping capital letters only
introduces unnecessary noise. Enabling option *lowercase* (True by default)
will lowercase all words in your documents.

Sometimes you want to keep things typed in all caps (e.g., abbreviations).
Setting *keepcaps* to True will do exactly that (default is False).

.. code:: python

    tokenizer = CrazyTokenizer(lowercase=True, keepcaps=True)
    tokenizer.tokenize('Moscow is the capital of RUSSIA!')

.. parsed-literal::

    ['moscow', 'is', 'the', 'capital', 'of', 'RUSSIA']

Normalizing
^^^^^^^^
Typing like thiiiis is amaaaaazing! However, in terms of text classification
*amaaaaazing* is probably not too different from *amaaaazing*. CrazyTokenizer
can normalize sequences of repeated characters for you. Just set *normalize*
argument to the integer number. This is the number of characters you want to keep.
Deafult value is 3.

.. code:: python

    tokenizer = CrazyTokenizer(normalize=3)
    tokenizer.tokenize('GOOOOOOOOO Patriots!!!!')

.. parsed-literal::

    ['gooo', 'patriots']

Ignoring quotes
^^^^^^^^
People often quote other comments or tweets, but it doesn't mean that they
endorse the original message. Removing the content of the quotes can help
you to get rid of that. Just set *ignorequotes* to True (False by deafult).

.. code:: python

    tokenizer = CrazyTokenizer(ignorequotes=True)
    tokenizer.tokenize('And then she said: "I voted for Donald Trump"')

.. parsed-literal::

    ['and', 'then', 'she', 'said']
