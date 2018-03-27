"""
SpacyTokenizer: SpaCy-based tokenizer with Twitter- and Reddit-specific features
Author: Evgenii Nikitin <e.nikitin@nyu.edu>
"""

from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.tokens import Token, Span
from spacy.tokenizer import Tokenizer
import re, sys, os, string
from math import log
#import multiprocessing as mp
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer
except ModuleNotFoundError:
    print('nltk could not be imported, some features will be unavailable')

Token.set_extension('transformed_text', default='')

pos_emojis = [u'😂', u'❤', u'♥', u'😍', u'😘', u'😊', u'👌', u'💕', u'👏', u'😁', u'☺', u'♡', u'👍', u'✌', u'😏', 
u'😉', u'🙌', u'😄']
neg_emojis = [u'😭', u'😩', u'😒', u'😔', u'😱']
neutral_emojis = [u'🙏']

normalize_re = re.compile(r"([a-zA-Z])\1\1+")

prefix_re = re.compile(r'''^[\[\("']''')
suffix_re = re.compile(r'''[\]\)"'?!.]$''')
def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search)

class SpacyTokenizer(object):
    def __init__(self, **kwargs):
        """
        SpaCy-based tokenizer with many useful Twitter- and Reddit-specific features

        Parameters
        ----------
        lowercase : bool, default: True
            Whether to lowercase words

        keepcaps: bool, deafult: True
            Whether to keep ALLCAPS WORDS

        normalize: int, default: 3
            Normalization of repeated charachers. The value of parameter
            determines the number of occurences to keep.
            Example: "awesoooooome" -> "awesooome"

        ignorequotes: bool, deafult: False
            Whether to remove everything in double and single quotes

        ignorestopwords: bool or list, deafult: False
            Whether to ignore stopwords
                True: attempt to get a list of stopwords from NLTK package,
                      language is determined by the 'language' parameter
                False: keep all words
                list: list of stopwords to remove

        keepwords: list, deafult: None
            List of words to keep. This should be used if you want to
            remove NLTK stopwords, but want to keep a few specific words.ignorequotes

        stem: {False, 'stem', 'lemm'}, deafult: False
            Word stemming
                False: do not perform word stemming
                'stem': use PorterStemmer from NLTK package
                'lemm': use WordNetLemmatizer from NLTK package

        removepunct: bool, default: True
            Whether to remove punctuation

        remove_nonunicode: bool, default: False
            Whether to remove all non-Unicode characters

        decontract: bool, deafult: False
            Whether to perform decontraction of the common contractions
            Example: "'ll" -> " will"

        splithashtags: bool, deafult: False
            Whether to perform hashtag splitting
            Example: "#vladimirputinisthebest" -> "vladimir putin is the best"

        twitter_handles, urls, hashtags, numbers, 
        subreddits, reddit_usernames, emails: None or str
            Replacement of the different types of tokens
                None: do not perform
                str: replacement token
                '': special case of the replacement token, removes all occurrences

        extra_patterns: None or list of tuples, default: None
            Replacement of any user-supplied extra patterns.
            It must be a list of tuples: (name, re_pattern, replacement_token):
                name (str): name of the pattern
                re_pattern (_sre.SRE_Pattern): compiled re pattern
                replacement_token (str): replacement token

        language: str, deafult: 'english'
            Main language of the documents

        pos_emojis, neg_emojis, neutral_emojis: None, True or list, deafult: None
            Whether to replace positive, negative, and neutral emojis with the special tokens
                None: do not perform replacement
                True: perform replacement of the default lists of emojis
                list: list of emojis to replace
        """

        self._default_values = dict(lowercase=True, keepcaps=True, normalize=3, ignorequotes=False, ignorestopwords=False, 
                                    keepwords=None, stem=False, removepunct=True, remove_nonunicode=False, decontract=False, 
                                    splithashtags=False, twitter_handles='TOKENTWITTERHANDLE', urls='TOKENURL', hashtags='TOKENHASHTAG', 
                                    numbers='TOKENNUMBER', subreddits='TOKENSUBREDDIT', reddit_usernames='TOKENREDDITOR', 
                                    emails='TOKENEMAIL', extra_patterns=None, language='english',
                                    pos_emojis=None, neg_emojis=None, neutral_emojis=None)

        self.nlp = English()
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.matcher = Matcher(self.nlp.vocab)
        for (prop, default) in self._default_values.items():
            setattr(self, prop, kwargs.get(prop, default))

        self._replacements = {}

        if (self.ignorestopwords == True) & ('nltk' in sys.modules):
            self._stopwords = stopwords.words(self.language)
            print('Stopwords are set to NLTK {} set'.format(self.language))
        elif self.ignorestopwords == False:
            self._stopwords = []
        else:
            if self.ignorestopwords is not list:
                raise TypeError("ignorestopwords should be a list of words or boolean")
            self._stopwords = self.ignorestopwords

        if self.keepwords is None:
            self.keepwords = []

        normalize_flag = lambda text: bool(normalize_re.match(text))
        TO_NORMALIZE = self.nlp.vocab.add_flag(normalize_flag)

        twitter_handle_flag = lambda text: bool(re.compile(r"@\w{1,15}").match(text))
        TWITTER_HANDLE = self.nlp.vocab.add_flag(twitter_handle_flag)

        reddit_username_flag = lambda text: bool(re.compile(r"u/\w{1,20}").match(text))
        REDDIT_USERNAME = self.nlp.vocab.add_flag(reddit_username_flag)

        subreddit_flag = lambda text: bool(re.compile(r"/r/\w{1,20}").match(text))
        SUBREDDIT = self.nlp.vocab.add_flag(subreddit_flag)

        hashtag_flag = lambda text: bool(re.compile(r"#\w+[\w'-]*\w+").match(text))
        HASHTAG = self.nlp.vocab.add_flag(hashtag_flag)

        quote_flag = lambda text: bool(re.compile(r"'").match(text))
        doublequote_flag = lambda text: bool(re.compile(r'"').match(text))
        QUOTE = self.nlp.vocab.add_flag(quote_flag)
        DOUBLEQUOTE = self.nlp.vocab.add_flag(doublequote_flag)

        stopword_flag = lambda text: bool((text in self._stopwords) & (text not in self.keepwords))
        STOPWORD = self.nlp.vocab.add_flag(stopword_flag)

        self.matcher.add('STOPWORD', self._remove_token, [{STOPWORD: True}])

        if self.ignorequotes:
            self.matcher.add('SINGLE_QUOTES', self._merge_and_remove, [{'ORTH': "'"}, {'OP': '+', 'IS_ASCII': True, 'QUOTE': False}, {'ORTH': "'"}])
            self.matcher.add('DOUBLE_QUOTES', self._merge_and_remove, [{'ORTH': '"'}, {'OP': '+', 'IS_ASCII': True, 'DOUBLEQUOTE': False}, {'ORTH': '"'}])

        if self.lowercase & (not self.keepcaps):
            self.matcher.add('LOWERCASE', self._lowercase, [{'IS_LOWER': False}])
        elif self.lowercase & self.keepcaps:
            self.matcher.add('LOWERCASE', self._lowercase, [{'IS_LOWER': False, 'IS_UPPER': False}])

        if isinstance(self.normalize, int):
            self.matcher.add('NORMALIZE', self._normalize, [{TO_NORMALIZE: True}])

        if self.removepunct:
            self.matcher.add('PUNCTUATION', self._remove_token, [{'IS_PUNCT': True}])


        if self.subreddits:
            self.matcher.add('SUBREDDIT', self._replace_token, [{SUBREDDIT: True}])
            self._replacements['SUBREDDIT'] = self.subreddits

        if self.twitter_handles:
            self.matcher.add('TWITTER_HANDLE', self._replace_token, [{TWITTER_HANDLE: True}])
            self._replacements['TWITTER_HANDLE'] = self.twitter_handles

        if self.reddit_usernames:
            self.matcher.add('REDDIT_USERNAME', self._replace_token, [{REDDIT_USERNAME: True}])
            self._replacements['REDDIT_USERNAME'] = self.reddit_usernames

        if self.hashtags:
            self.matcher.add('HASHTAG', self._replace_token, [{HASHTAG: True}])
            self._replacements['HASHTAG'] = self.hashtags

        if self.urls:
            self.matcher.add('URL', self._replace_token, [{'LIKE_URL': True}])
            self._replacements['URL'] = self.urls

        if self.numbers:
            self.matcher.add('NUMBER', self._replace_token, [{'LIKE_NUM': True}])
            self._replacements['NUMBER'] = self.numbers

        if self.emails:
            self.matcher.add('EMAIL', self._replace_token, [{'LIKE_EMAIL': True}])
            self._replacements['EMAIL'] = self.emails


        if self.pos_emojis:
            if isinstance(self.pos_emojis, list) == False:
                self.pos_emojis = pos_emojis
            pos_patterns = [[{'ORTH': emoji}] for emoji in self.pos_emojis]
            self.matcher.add('HAPPY', self._replace_token, *pos_patterns)
            self._replacements['HAPPY'] = 'POS_EMOJI'

        if self.neg_emojis:
            if isinstance(self.neg_emojis, list) == False:
                self.neg_emojis = neg_emojis
            neg_patterns = [[{'ORTH': emoji}] for emoji in self.neg_emojis]
            self.matcher.add('SAD', self._replace_token, *neg_patterns)
            self._replacements['SAD'] = 'NEG_EMOJI'

        if self.neutral_emojis:
            if isinstance(self.neutral_emojis, list) == False:
                self.neutral_emojis = neutral_emojis
            neutral_patterns = [[{'ORTH': emoji}] for emoji in self.neutral_emojis]
            self.matcher.add('NEUTRAL', self._replace_token, *neutral_patterns)
            self._replacements['NEUTRAL'] = 'NEUTRAL_EMOJI'

        if isinstance(self.extra_patterns, list):
            self.flags = {}
            for extra_pattern in self.extra_patterns:
                name, re_pattern, replacement_token = extra_pattern
                flag = lambda text: bool(re_pattern.match(text))
                self.flags[name] = self.nlp.vocab.add_flag(flag)
                self.matcher.add(name, self._replace_token, [{self.flags[name]: True}])
                self._replacements[name] = replacement_token

        if (self.stem) and ('nltk' in sys.modules):
            if self.stem == 'stem':
                self.stemmer = PorterStemmer()
            elif self.stem == 'lemm':
                self.stemmer = WordNetLemmatizer()
            else:
                raise Exception('Stemming method {} is not supported'.format(self.stem))
            self.matcher.add('WORD_TO_STEM', self._stem_word, [{'IS_ALPHA': True}])

        if (self.splithashtags) & (not self.hashtags):
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                os.path.join('data', 'wordsfreq.txt'))
            self.words = open(file).read().split()
            self.wordcost = dict((k, log((i+1)*log(len(self.words)))) for i,k in enumerate(self.words))
            self.maxword = max(len(x) for x in self.words)
            self.matcher.add('HASHTAG', self._split_hashtags, [{HASHTAG: True}])

    def _lowercase(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = tok._.transformed_text.lower()

    def _stem_word(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            if self.stem == 'stem':
                tok._.transformed_text = self.stemmer.stem(tok._.transformed_text)
            elif self.stem == 'lemm':
                tok._.transformed_text = self.stemmer.lemmatize(tok._.transformed_text)

    def _normalize(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = normalize_re.sub(r"\1"*self.normalize, tok._.transformed_text)

    def _replace_token(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]
        replacement_token = self._replacements[doc.vocab.strings[match_id]]
        for tok in span:
            tok._.transformed_text = replacement_token

    def _remove_token(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = ''

    def _split_hashtags(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            poss = self._infer_spaces(tok.text[1:]).split()
            if len(poss) > 0:
                tok._.transformed_text = poss

    def _merge_and_remove(self, matcher, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]
        span.merge()
        for tok in span:
            tok._.transformed_text = ''

    def _decontract(self, sentence):
        sentence = re.sub(r"won't", "will not", sentence)
        sentence = re.sub(r"can\'t", "can not", sentence)
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'s", " is", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'t", " not", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'m", " am", sentence)
        return sentence

    def _infer_spaces(self, s):
        """
        Infer the location of spaces
        """
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        def best_match(i):
            """
            Find the best match for the first i characters,
            assuming cost has been built for the first i-1 characters

            Returns:
                match_cost, match_length
        """
            candidates = enumerate(reversed(cost[max(0, i-self.maxword):i]))
            return min((c + self.wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

        #build the cost array
        cost = [0]
        for i in range(1, len(s) + 1):
            c, k = best_match(i)
            cost.append(c)

        #backtrack to recover the minimial-cost string
        out = []
        i = len(s)
        while i > 0:
            c, k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])
            i -= k

        return " ".join(reversed(out))


    def tokenize_doc(self, text):
        """
        Tokenize document
        Parameters
        ----------
        text: str
            Document to tokenize
        """        
        if self.remove_nonunicode:
            try:
                text = text.encode('utf-8').decode('unicode-escape')
                text = ''.join(filter(lambda x: x in string.printable, text))
            except UnicodeDecodeError:
                pass

        if self.decontract:
            text = self._decontract(text)

        doc = self.nlp(text)
        for t in doc:
            t._.transformed_text = t.text

        matches = self.matcher(doc)

        tokens = []
        for t in doc:
            if isinstance(t._.transformed_text, list):
                tokens.extend(t._.transformed_text)
            elif t._.transformed_text != '':
                tokens.append(t._.transformed_text)
        return tokens

    def tokenize_docs(self, texts, cores=2):
        """
        DO NOT USE
        """
        pool = mp.Pool(processes=cores)
        results = [pool.apply(self.tokenize_doc, args=(x,)) for x in texts]
        return results

