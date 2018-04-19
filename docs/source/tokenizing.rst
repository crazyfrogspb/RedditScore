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

CrazyTokenizer is based on the amazing `spaCY NLP framework <https://spacy.io/>`__.
Make sure to check it out!

Initializing
--------------------
To import and to initialize an instance of CrazyTokenizer with the default
preprocessing options, do the following:

>>> from redditscore.tokenizer import CrazyTokenizer
>>> tokenizer = CrazyTokenizer()

Now you can start tokenizing!

>>> text = ("@crazyfrogspb Hey,dude, have you heard that"
>>>         " https://github.com/crazyfrogspb/RedditScore is the best Python library?")
>>> tokenizer.tokenizer(text)
['@crazyfrogspb', 'hey', 'dude', 'have', 'you', 'heard', 'that',
'https://github.com/crazyfrogspb/RedditScore', 'is', 'the', 'best', 'python', 'library']

Features
--------------------

Lowercasing and all caps
^^^^^^^^
For many text classification problems, keeping capital letters only
introduces unnecessary noise. Setting ``lowercase=True`` (True by default)
will lowercase all words in your documents.

Sometimes you want to keep things typed in all caps (e.g., abbreviations).
Setting ``keepcaps=True`` will do exactly that (default is False).

>>> tokenizer = CrazyTokenizer(lowercase=True, keepcaps=True)
>>> tokenizer.tokenize('Moscow is the capital of RUSSIA!')
['moscow', 'is', 'the', 'capital', 'of', 'RUSSIA']

Normalizing
^^^^^^^^
Typing like thiiiis is amaaaaazing! However, in terms of text classification
*amaaaaazing* is probably not too different from *amaaaazing*. CrazyTokenizer
can normalize sequences of repeated characters for you. Just set ``normalize=n``,
where *n* is the number of characters you want to keep. Default value is 3.

>>> tokenizer = CrazyTokenizer(normalize=3)
>>> tokenizer.tokenize('GOOOOOOOOO Patriots!!!!')
['gooo', 'patriots']

Ignoring quotes
^^^^^^^^
People often quote other comments or tweets, but it doesn't mean that they
endorse the original message. Removing the content of the quotes can help
you to get rid of that. Just set ``ignorequotes=True`` (False by deafult).

>>> tokenizer = CrazyTokenizer(ignorequotes=True)
>>> tokenizer.tokenize('And then she said: "I voted for Donald Trump"')
['and', 'then', 'she', 'said']

Removing stop words
^^^^^^^^
Removing stop words can sometimes significantly boost performance of your
classifier. CrazyTokenizer gives you a few options to remove stop words:

  - Using built-in list of the english stop words (``ignorestopwords=True``)

  >>> tokenizer = CrazyTokenizer(ignorestopwords=True)
  >>> tokenizer.tokenize('PhD life is great: eat, work, and sleep')
  ['phd', 'life', 'great', 'eat', 'work', 'sleep']

  - Using NLTK lists of stop words. Just pass the name of the language
    of your documents to the ``ignorestopwords`` parameter.

  >>> tokenizer = CrazyTokenizer(ignorestopwords='english')
  # You might have to run nltk.download('stopwords') first
  >>> tokenizer.tokenize('PhD life is great: eat, work, and sleep')
  ['phd', 'life', 'great', 'eat', 'work', 'sleep']

  - Alternatively, you can supply your own custom list of the stop words. Letter case doesn't matter.

  >>> tokenizer = CrazyTokenizer(ignorestopwords=['Vladimir', "Putin"])
  >>> tokenizer.tokenize("The best leader in the world is Vladimir Putin")
  ['the', 'best', 'leader', 'in', 'the', 'world', 'is']

Word stemming and lemmatizing
^^^^^^^^
If you have NLTK installed, CrazyTokenizer can use PorterStemmer or
WordNetLemmatizer for you. Just pass ``stem`` or ``lemm`` options
respectively to ``stem`` parameter.

>>> tokenizer = CrazyTokenizer(stem='stem')
>>> tokenizer.tokenize("I am an unbelievably fantastic human being")
['i', 'am', 'an', 'unbeliev', 'fantast', 'human', 'be']

Removing punctuation and linebreaks
^^^^^^^^
Punctuation and linebreak characters usually just introduce extra noise
to your text classification problem,
so you can easily remove it with ``removepunct`` and ``removebreaks`` options.
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
CrazyTokenizer correctly handles Twitter handles, subreddits, Reddit usernames,
emails, all sorts of numbers, and extracts them as separate tokens:

>>> tokenizer = CrazyTokenizer()
>>> text = "@crazyfrogspb recommends /r/BeardAdvice!"
>>> tokenizer.tokenize(text)
['@crazyfrogspb', 'recommends', '/r/beardadvice']

However, you might want to completely remove certain types of tokens
(for example, it makes sense to remove subreddit names if you want to compute
RedditScores for the Twitter data), or to replace them with special tokens.
Well, it's your lucky day, CrazyTokenizer can do that!

>>> tokenizer = CrazyTokenizer(subreddits='', twitter_handles='ANOTHER_TWITTER_USER')
>>> tokenizer.tokenize(text)
['ANOTHER_TWITTER_USER', 'recommends']

There is a special option for Twitter handles: 'realname'. It replaces each
handle with the screen name of the user that is listed in their profile.

>>> tokenizer = CrazyTokenizer(splithashtags=True, twitter_handles='realname')
>>> tokenizer.tokenize('@realDonaldTrump please #MakeAmericaGreatAgain')
['donald', 'j.', 'trump', 'please', 'make', 'america', 'great', 'again']

URLs
^^^^^^^^
NLP practicioners often simply remove all URL occurrences since they do not
seem to contain any useful semantic information. Of course, CrazyTokenizer
correctly recognizes URLs as separate tokens and can remove or replace them
with a placeholder token.

>>> tokenizer = CrazyTokenizer(urls=False)
>>> text = "Where is my job then?https://t.co/pN2TE5HDQm"
>>> tokenizer.tokenize(text)
['where', 'is', 'my', 'job', 'then', 'https://t.co/pN2TE5HDQm']
>>> tokenizer = CrazyTokenizer(urls='URL')
>>> tokenizer.tokenize(text)
['where', 'is', 'my', 'job', 'then', 'URL']

CrazyTokenizer can do something even more interesting though. Let's explore
all options one by one.

First, CrazyTokenizer can extract domains from your URLs.

>>> tokenizer = CrazyTokenizer(urls='domain')
>>> text = "http://nytimes.com or http://breitbart.com, that is the question"
>>> tokenizer.tokenize(text)
['nytimes', 'or', 'breitbart', 'that', 'is', 'the', 'question']

Unfortunately, links on Twitter are often shortened, so extracting domain
directly doesn't make a lot of sense. Not to worry though, CrazyTokenizer
can handle that for you! Setting ``urls='domain_unwrap_fast'`` will deal with
links shortened by the following URL shorteners:
t.co, bit.ly, goo.gl, tinyurl.com.

>>> tokenizer = CrazyTokenizer(urls='domain_unwrap_fast')
>>> text = "Where is my job then?https://t.co/pN2TE5HDQm"
>>> tokenizer.tokenize(text)
['where', 'is', 'my', 'job', 'then', 'bloomberg_domain']

If you want, CrazyTokenizer can attempt to unwrap ALL extracted URLs.

>>> tokenizer = CrazyTokenizer(urls='domain_unwrap')
>>> text = "Where is my job then?https://t.co/pN2TE5HDQm"
>>> tokenizer.tokenize(text)
['where', 'is', 'my', 'job', 'then', 'bloomberg_domain']

Last but not least, CrazyTokenizer can extract web page titles, tokenize them,
and insert to your tokenized sentences. Note: it won't extract titles from the
Twitter pages in order to avoid duplicating tweets content.

>>> tokenizer = CrazyTokenizer(urls='title')
>>> text = "I love Russia https://goo.gl/3ioXU4"
>>> tokenizer.tokenize(text)
['i', 'love', 'russia', 'russia', 'to', 'block', 'telegram', 'app', 'over', 'encryption', 'bbc', 'news']

**Please note** that CrazyTokenizer has to make requests to the websites,
and it is a very time-consuming operation, so CrazyTokenizer saves all
parsed domains and web page titles. If you plan to experiment with
the different preprocessing options and/or models, you should consider saving
extracted domains/titles and then supplying saved dictionary as an argument
to ``urls`` parameter.

>>> import json
>>> with open('domains.json', 'w') as f:
      json.dump(tokenizer._domains, f)
>>> with open('titles.json', 'w') as f:
      json.dump(tokenizer._titles, f)
>>> with open('domains.json', 'r') as f:
      domains = json.load(f)
>>> tokenizer = CrazyTokenizer(urls='title')
>>> with open('titles.json', 'r') as f:
      titles = json.load(f)
>>> tokenizer = CrazyTokenizer(urls=domains)
>>> >>> tokenizer = CrazyTokenizer(urls=titles)

Extra patterns and keeping untokenized
^^^^^^^^
You can also supply your own replacement rules to CrazyTokenizer. In particular,
you need to provide a tuple that contains unique name for your rule, compiled
re pattern and a replacement token.

Also, it makes sense to keep some common expressions (e.g., "New York Times")
untokenized. If you think that it can improve your model quality, feel free to
supply a list of strings that should be kept as single tokens.

>>> import re
>>> rule0 = re.compile(r"[S,s]ucks")
>>> rule1 = re.compile(r"[R,r]ules")
>>> tokenizer = CrazyTokenizer(extra_patterns=[('rule0', rule0, 'rules'),
                                               ('rule1', 'rule1, "sucks')],
                               keep_untokenized=['St.Petersburg'],
                               lowercase=False)
>>> text = "Moscow rules, St.Petersburg sucks"
['Moscow', 'sucks', 'St.Petersburg', 'rules']

Converting whitespaces to underscores
^^^^^^^^
Popular implementations of models (most notably, fastText) do not support
custom token splitting rules and simply split on whitespaces. In order to deal
with that, CrazyTokenizer can replace all whitespaces in the final tokens by
underscores (enabled by deafult).

>>> tokenizer = CrazyTokenizer(whitespaces_to_underscores=True, keep_untokenized=["New York"])
>>> text = "New York is a great place to make a rat friend"
>>> tokenizer.tokenize(text)
['new_york', 'is', 'a', 'great', 'place', 'to', 'make', 'a', 'rat', 'friend']

Removing non-unicode characters
^^^^^^^^
>>> tokenizer = CrazyTokenizer(remove_nonunicode=True)
>>> text = "Россия - священная наша держава, Россия - великая наша страна!"
>>> tokenizer.tokenize(text)
[]

Emojis
^^^^^^^^
Social media users are notoriously famous for their excessive use of emojis.
CrazyTokenizer correctly separates consecutive emojis.

In addition, CrazyTokenizer can replace different kind of emojis with the
corresponding word tokens.

>>> tokenizer = CrazyTokenizer(pos_emojis=True, neg_emojis=True, neutral_emojis=True)
>>> text = '😍😭😩???!!!!'
>>> tokenizer.tokenize(text)
['POS_EMOJI', 'NEG_EMOJI', 'NEG_EMOJI']

You can supply your own lists of emojis as well.

>>> tokenizer = CrazyTokenizer(pos_emojis=['🌮', '🍔'], neutral_emojis=['😕'], removepunct=False)
>>> text = '🌮 + 🍔 = 😕'
>>> tokenizer.tokenize(text)
['POS_EMOJI', '+', 'POS_EMOJI', '=', 'NEUTRAL_EMOJI']

Unicode and hex characters
^^^^^^^^
Sometimes your data gets messed up as a result of repeated save/load operations.
If your data contains a lot of substrings that look like this: ``\\xe2\\x80\\x99``
or this: ``U+1F601``, try setting ``latin_chars_fix=True``.

>>> tokenizer = CrazyTokenizer(latin_chars_fix=True)
>>> s = "I\\xe2\\x80\\x99m so annoyed by these characters \\xF0\\x9F\\x98\\xA2"

Tokenizing a bunch of documents
--------------------
Tokenizing 10,000 Reddit comments takes about 10 seconds on my Gigabyte Aero 15.

>>> import pandas as pd
>>> import os
>>> df = pd.read_csv(os.path.join('redditscore', 'data', 'reddit_small_sample.csv'))
>>> tokenizer = CrazyTokenizer(urls='domain')
>>> df['tokens'] = df['body'].apply(tokenizer.tokenize)
