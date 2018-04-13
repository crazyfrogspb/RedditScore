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

>>> from redditscore.tokenizer import CrazyTokenizer
>>> tokenizer = CrazyTokenizer()

Now you can start tokenizing!

>>> text = "@crazyfrogspb Hey,dude, have you heard that https://github.com/crazyfrogspb/RedditScore is the best Python library?"
>>> tokenizer.tokenizer(text)
['TOKENTWITTERHANDLE', 'hey', 'dude', 'have', 'you', 'heard', 'that', 'github_domain', 'is', 'the', 'best', 'python', 'library']

Features
--------------------

Lowercasing and all caps
^^^^^^^^
For many text classification problems, keeping capital letters only
introduces unnecessary noise. Enabling option *lowercase* (True by default)
will lowercase all words in your documents.

Sometimes you want to keep things typed in all caps (e.g., abbreviations).
Setting *keepcaps* to True will do exactly that (default is False).

>>> tokenizer = CrazyTokenizer(lowercase=True, keepcaps=True)
>>> tokenizer.tokenize('Moscow is the capital of RUSSIA!')
['moscow', 'is', 'the', 'capital', 'of', 'RUSSIA']

Normalizing
^^^^^^^^
Typing like thiiiis is amaaaaazing! However, in terms of text classification
*amaaaaazing* is probably not too different from *amaaaazing*. CrazyTokenizer
can normalize sequences of repeated characters for you. Just set *normalize*
argument to the integer number. This is the number of characters you want to keep.
Deafult value is 3.

>>> tokenizer = CrazyTokenizer(normalize=3)
>>> tokenizer.tokenize('GOOOOOOOOO Patriots!!!!')
['gooo', 'patriots']

Ignoring quotes
^^^^^^^^
People often quote other comments or tweets, but it doesn't mean that they
endorse the original message. Removing the content of the quotes can help
you to get rid of that. Just set *ignorequotes* to True (False by deafult).

>>> tokenizer = CrazyTokenizer(ignorequotes=True)
>>> tokenizer.tokenize('And then she said: "I voted for Donald Trump"')
['and', 'then', 'she', 'said']

Removing stop words
^^^^^^^^
Removing stop words can sometimes significantly boost performance of your
classifier. CrazyTokenizer gives you a few options to remove stop words:

  - Using NLTK lists of stop words. Just pass the name of the language
    of your documents to the *ignorestopwords* parameter.

  >>> tokenizer = CrazyTokenizer(ignorestopwords='english')
  # You might have to run nltk.download('stopwords') first
  >>> tokenizer.tokenize('PhD life is great: eat, work, and sleep')
  ['phd', 'life', 'great', 'eat', 'work', 'sleep']

  - Alternatively, you can supply your own custom list of the stop words.
  Letter case doesn't matter.

  >>> tokenizer = CrazyTokenizer(ignorestopwords=['Vladimir', "Putin"])
  >>> tokenizer.tokenize("The best leader in the world is Vladimir Putin")
  ['the', 'best', 'leader', 'in', 'the', 'world', 'is']

Word stemming and lemmatizing
^^^^^^^^
If you have NLTK installed, CrazyTokenizer can use PorterStemmer or
WordNetLemmatizer for you. Just pass 'stem' or 'lemm' options
respectively to *stem* parameter.

>>> tokenizer = CrazyTokenizer(stem='stem')
>>> tokenizer.tokenize("I am an unbelievably fantastic human being")
['i', 'am', 'an', 'unbeliev', 'fantast', 'human', 'be']

Removing punctuation and lineb
^^^^^^^^
Punctuation and linebreak characters usually just introduce extra noise
to your text classification problem,
so you can easily remove it with *removepunct* and *removebreaks* options.
Both default to True.

>>> tokenizer = CrazyTokenizer(removepunct=True, removebreaks=True)
>>> tokenizer.tokenize("I love my life, friends, and oxford commas. \n Amen!")
['i', 'love', 'my', 'life', 'friends', 'and', 'oxford', 'commas', 'amen']

Decontracting
^^^^^^^^
CrazyTokenizer can attempt to expand some of those annoying contractions
for you. **Note**: use at your own risk.

>>> tokenizer = CrazyTokenizer(decontract=True)
>>> tokenizer.tokenize("I'll have two number nines, a number nine large...")
['i', 'will', 'have', 'two', 'number', 'nines', 'a', 'number', 'nine', 'large']

Dealing with hashtags
^^^^^^^^
Hashtags are super-popular on Twitter. CrazyTokenizer can do one of
three things about them:

  - Do nothing (``hashtags=False, splithashtags=False``)
  - Replace all of them with a placeholder token (``hashtags='TOKEN'``)
  - Split them into separate words (``hashtags=False, splithashtags=True``)

Splitting hashtags is especially useful for the Reddit-based models since
hashtags are not used on Reddit, and you can potentially lose a lot of semantic
information when you calculate RedditScores for the Twitter data.

>>> tokenizer = CrazyTokenizer(hashtags=False, splithashtags=False)
>>> text = "Let's #makeamericagreatagain#americafirst"
>>> tokenizer.tokenize(text)
["let's", "#makeamericagreatagain", "#americafirst"]
>>> tokenizer = CrazyTokenizer(hashtags="HASHTAG_TOKEN", splithashtags=False)
["let's", "HASHTAG_TOKEN", "HASHTAG_TOKEN"]
>>> tokenizer = CrazyTokenizer(hashtags=False, splithashtags=True)
["let's", "make", "america", "great", "again", "america", "first"]

Dealing with special tokens
^^^^^^^^
CrazyTokenizer correctly handles twitter_handles, subreddits, reddit_usernames,
emails, all forms of numbers, and splits them as separate tokens:
>>> tokenizer = CrazyTokenizer()
>>> text = "@crazyfrogspb recommends /r/BeardAdvice!"
>>> tokenizer.tokenize(text)
['@crazyfrogspb', 'recommends', '/r/beardadvice']

However, you might want to completely remove certain types of tokens
(for example, it makes to remove subreddit names if you want to compute
RedditScores for the Twitter data), or to replace them with special tokens.
Well, it's your lucky day, CrazyTokenizer can do that!

>>> tokenizer = CrazyTokenizer(subreddits='', twitter_handles='ANOTHER_TWITTER_USER')
>>> tokenizer.tokenize(text)
['ANOTHER_TWITTER_USER', 'recommends']

URLs
^^^^^^^^
