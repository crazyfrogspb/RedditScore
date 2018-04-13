CrazyTokenizer
=====================

Tokenizer description
--------------------

CrazyTokenizer is a part of `RedditScore project <https://github.com/crazyfrogspb/RedditScore>`__.
It's a tokenizer - tool for splitting strings of text into tokens. Tokens can
then be used as input for a variety of machine learning models.
CrazyTokenizer was developed specifically for tokenizing Reddit comments and
tweets, and it includes many features to deal with these types of documents.
Of course, you can use for any other kind of text data as well!

**Note:** RedditScore library and this tutorial are work-in-progress.
`Let me know if you experience any issues <https://github.com/crazyfrogspb/RedditScore/issues>`__.

Initializing
--------------------
To import and to initialize tokenizer with the default preprocessing options,
do the following:

from redditscore.tokenizer import CrazyTokenizer
tokenizer = CrazyTokenizer()

.. code:: python

    from redditscore.tokenizer import CrazyTokenizer
    tokenizer = CrazyTokenizer()

Cool
