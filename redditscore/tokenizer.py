"""
SpacyTokenizer: SpaCy-based tokenizer with Twitter- and Reddit-specific features
Author: Evgenii Nikitin <e.nikitin@nyu.edu>
"""

import re
import sys
import os
import string
import warnings
from math import log
from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.tokens import Token, Doc
from spacy.tokenizer import Tokenizer
import tldextract
import requests
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer
except ModuleNotFoundError:
    print('nltk could not be imported, some features will be unavailable')

Token.set_extension('transformed_text', default='')
Doc.set_extension('tokens', default='')

POS_EMOJIS = [u'😂', u'❤', u'♥', u'😍', u'😘', u'😊', u'👌', u'💕', u'👏', u'😁', u'☺', u'♡', u'👍', u'✌', u'😏',
              u'😉', u'🙌', u'😄']
NEG_EMOJIS = [u'😭', u'😩', u'😒', u'😔', u'😱']
NEUTRAL_EMOJIS = [u'🙏']

NORMALIZE_RE = re.compile(r"([a-zA-Z])\1\1+")
URLS_RE = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

PREFIX_RE = re.compile(r'''^[\[\("'\s]''')
SUFFIX_RE = re.compile(r'''[\]\)"'?!.\s,:;]$''')


def custom_tokenizer(nlp):
    """
    Change default SpaCy tokenizing rules
    """
    return Tokenizer(nlp.vocab, prefix_search=PREFIX_RE.search,
                     suffix_search=SUFFIX_RE.search)


def true_or_empty(arg):
    """
    Check whether argument is True or an empty string
    """
    if arg or (arg == ''):
        return True
    return False


class CrazyTokenizer(object):
    def __init__(self, lowercase=True, keepcaps=True, normalize=3, ignorequotes=False,
                 ignorestopwords=False, keepwords=None, stem=False, removepunct=True,
                 removebreaks=True, remove_nonunicode=False, decontract=False,
                 splithashtags=True, twitter_handles='TOKENTWITTERHANDLE',
                 urls='domain_unwrap', hashtags=False, numbers='TOKENNUMBER',
                 subreddits='TOKENSUBREDDIT', reddit_usernames='TOKENREDDITOR',
                 emails='TOKENEMAIL', extra_patterns=None, keep_untokenized=None,
                 pos_emojis=None, neg_emojis=None, neutral_emojis=None):
        """
        Initialize CrazyTokenizer object

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
            Whether to remove everything in double quotes

        ignorestopwords: str, False or list, deafult: False
            Whether to ignore stopwords
                str: language to get a list of stopwords for from NLTK package
                False: keep all words
                list: list of stopwords to remove (all words must be lowercased)

        keepwords: list, deafult: None
            List of words to keep. This should be used if you want to
            remove NLTK stopwords, but want to keep a few specific words.
            All words must be lowercased.

        stem: {False, 'stem', 'lemm'}, deafult: False
            Word stemming
                False: do not perform word stemming
                'stem': use PorterStemmer from NLTK package
                'lemm': use WordNetLemmatizer from NLTK package

        removepunct: bool, default: True
            Whether to remove punctuation

        removebreaks: bool, default: True
            Whether to remove linebreak charachters

        remove_nonunicode: bool, default: False
            Whether to remove all non-Unicode characters

        decontract: bool, deafult: False
            Whether to perform decontraction of the common contractions
            Example: "'ll" -> " will"

        splithashtags: bool, deafult: True
            Whether to perform hashtag splitting
            Example: "#vladimirputinisthebest" -> "vladimir putin is the best"

        twitter_handles, hashtags, numbers,
        subreddits, reddit_usernames, emails: None or str
            Replacement of the different types of tokens
                None: do not perform
                str: replacement token
                '': special case of the replacement token, removes all occurrences

        urls: None or str, default: 'domain_unwrap'
            Replacement of parsed URLs
                None: do not perform
                str: replacement token
                '': special case of the replacement token, removes all occurrences
                'domain': extract domain
                'domain_unwrap': extract domain after 'unwwraping links like t.co/'

        extra_patterns: None or list of tuples, default: None
            Replacement of any user-supplied extra patterns.
            It must be a list of tuples: (name, re_pattern, replacement_token):
                name (str): name of the pattern
                re_pattern (_sre.SRE_Pattern): compiled re pattern
                replacement_token (str): replacement token

        keep_untokenized: None or list, deafault: None
            List of expressions to keep untokenized
            Example: ["New York", "Los Angeles", "San Francisco"]

        language: str, deafult: 'english'
            Main language of the documents

        pos_emojis, neg_emojis, neutral_emojis: None, True or list, deafult: None
            Whether to replace positive, negative, and neutral emojis with the special tokens
                None: do not perform replacement
                True: perform replacement of the default lists of emojis
                list: list of emojis to replace
        """
        self.nlp = English()
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.merging_matcher = Matcher(self.nlp.vocab)
        self.matcher = Matcher(self.nlp.vocab)
        self.tokenizing_parameters = locals()
        self.remove_nonunicode = remove_nonunicode
        self.decontract = decontract
        self._replacements = {}

        if isinstance(ignorestopwords, str) and ('nltk' in sys.modules):
            try:
                self._stopwords = stopwords.words(ignorestopwords)
            except OSError:
                raise ValueError('Language {} was not found by NLTK'.format(ignorestopwords))
        elif isinstance(ignorestopwords, list):
            self._stopwords = ignorestopwords
        elif (not ignorestopwords) or (ignorestopwords is None):
            self._stopwords = []
        else:
            raise TypeError('Type {} is not supported by ignorestopwords parameter'.format(type(ignorestopwords)))

        if keepwords is None:
            keepwords = []

        if lowercase and (not keepcaps):
            self.matcher.add('LOWERCASE', self._lowercase, [{'IS_LOWER': False}])
        elif lowercase and keepcaps:
            self.matcher.add('LOWERCASE', self._lowercase, [{'IS_LOWER': False, 'IS_UPPER': False}])

        if removepunct:
            self.matcher.add('PUNCTUATION', self._remove_token, [{'IS_PUNCT': True}])

        if removebreaks:
            break_check = lambda text: bool(re.compile(r"[\r\n]+").fullmatch(text))
            break_flag = self.nlp.vocab.add_flag(break_check)
            self.matcher.add('BREAK', self._remove_token, [{break_flag: True}])

        if normalize:
            normalize_check = lambda text: bool(NORMALIZE_RE.search(text))
            normalize_flag = self.nlp.vocab.add_flag(normalize_check)
            self.matcher.add('NORMALIZE', self._normalize, [{normalize_flag: True}])

        if true_or_empty(numbers):
            number_check = lambda text: bool(re.compile(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?").fullmatch(text))
            number_flag = self.nlp.vocab.add_flag(number_check)
            self.matcher.add('NUMBER', self._replace_token, [{number_flag: True}])
            self._replacements['NUMBER'] = numbers

        if true_or_empty(urls):
            if urls == 'domain':
                self.matcher.add('URL', self._extract_domain, [{'LIKE_URL': True}])
            elif urls == 'domain_unwrap':
                self.matcher.add('URL', self._unwrap_domain, [{'LIKE_URL': True}])
            else:
                self.matcher.add('URL', self._replace_token, [{'LIKE_URL': True}])
                self._replacements['URL'] = urls

        if true_or_empty(twitter_handles):
            twitter_handle_check = lambda text: bool(re.compile(r"@\w{1,15}").fullmatch(text))
            twitter_handle_flag = self.nlp.vocab.add_flag(twitter_handle_check)
            self.matcher.add('TWITTER_HANDLE', self._replace_token, [{twitter_handle_flag: True}])
            self._replacements['TWITTER_HANDLE'] = twitter_handles

        if true_or_empty(reddit_usernames):
            reddit_username_check = lambda text: bool(re.compile(r"u/\w{1,20}").fullmatch(text))
            reddit_username_flag = self.nlp.vocab.add_flag(reddit_username_check)
            self.matcher.add('REDDIT_USERNAME', self._replace_token, [{reddit_username_flag: True}])
            self._replacements['REDDIT_USERNAME'] = reddit_usernames

        if true_or_empty(subreddits):
            subreddit_check = lambda text: bool(re.compile(r"/r/\w{1,20}").fullmatch(text))
            subreddit_flag = self.nlp.vocab.add_flag(subreddit_check)
            self.matcher.add('SUBREDDIT', self._replace_token, [{subreddit_flag: True}])
            self._replacements['SUBREDDIT'] = subreddits

        if true_or_empty(hashtags) or splithashtags:
            hashtag_check = lambda text: bool(re.compile(r"#\w+[\w'-]*\w+").fullmatch(text))
            hashtag_flag = self.nlp.vocab.add_flag(hashtag_check)
            self.matcher.add('HASHTAG', self._hashtag_postprocess, [{hashtag_flag: True}])
            file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                os.path.join('data', 'wordsfreq.txt'))
            self.words = open(file).read().split()
            self.wordcost = dict((k, log((i+1)*log(len(self.words)))) for i, k in enumerate(self.words))
            self.maxword = max(len(x) for x in self.words)

        if true_or_empty(emails):
            self.matcher.add('EMAIL', self._replace_token, [{'LIKE_EMAIL': True}])
            self._replacements['EMAIL'] = emails

        if ignorequotes:
            self.merging_matcher.add('QUOTE', None, [{'ORTH': '"'}, {'OP': '*', 'IS_ASCII': True}, {'ORTH': '"'}])
            doublequote_check = lambda text: bool(re.compile(r'^".*"$').fullmatch(text))
            doublequote_flag = self.nlp.vocab.add_flag(doublequote_check)
            self.matcher.add('DOUBLE_QUOTES', self._remove_token, [{doublequote_flag: True}])

        if self._stopwords:
            stopword_check = lambda text: bool((text.lower() in self._stopwords) & (text.lower() not in keepwords))
            stopword_flag = self.nlp.vocab.add_flag(stopword_check)
            self.matcher.add('STOPWORD', self._remove_token, [{stopword_flag: True}])

        self.merging_matcher.add('HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])
        self.merging_matcher.add('SUBREDDIT', None, [{'ORTH': '/r'}, {'ORTH': '/'}, {'IS_ASCII': True}])
        self.merging_matcher.add('REDDIT_USERNAME', None, [{'ORTH': 'u'}, {'ORTH': '/'}, {'IS_ASCII': True}])

        if keep_untokenized is not None:
            for i, phrase in enumerate(keep_untokenized):
                phrase_tokens = phrase.split(' ')
                rule = []
                for token in phrase_tokens:
                    rule.append({'ORTH': token})
                self.merging_matcher.add('RULE_' + str(i), None, rule)

        if pos_emojis:
            if not isinstance(pos_emojis, list):
                pos_emojis = POS_EMOJIS
            pos_patterns = [[{'ORTH': emoji}] for emoji in pos_emojis]
            self.matcher.add('HAPPY', self._replace_token, *pos_patterns)
            self._replacements['HAPPY'] = 'POS_EMOJI'

        if neg_emojis:
            if not isinstance(neg_emojis, list):
                neg_emojis = NEG_EMOJIS
            neg_patterns = [[{'ORTH': emoji}] for emoji in neg_emojis]
            self.matcher.add('SAD', self._replace_token, *neg_patterns)
            self._replacements['SAD'] = 'NEG_EMOJI'

        if neutral_emojis:
            if not isinstance(neutral_emojis, list):
                neutral_emojis = NEUTRAL_EMOJIS
            neutral_patterns = [[{'ORTH': emoji}] for emoji in neutral_emojis]
            self.matcher.add('NEUTRAL', self._replace_token, *neutral_patterns)
            self._replacements['NEUTRAL'] = 'NEUTRAL_EMOJI'

        if isinstance(extra_patterns, list):
            self.flags = {}
            for name, re_pattern, replacement_token in extra_patterns:
                flag = lambda text: bool(re_pattern.match(text))
                self.flags[name] = self.nlp.vocab.add_flag(flag)
                self.matcher.add(name, self._replace_token, [{self.flags[name]: True}])
                self._replacements[name] = replacement_token

        if stem and ('nltk' in sys.modules):
            self._stem = stem
            if stem == 'stem':
                self.stemmer = PorterStemmer()
            elif stem == 'lemm':
                self.stemmer = WordNetLemmatizer()
            else:
                raise ValueError('Stemming method {} is not supported'.format(stem))
            self.matcher.add('WORD_TO_STEM', self._stem_word, [{'IS_ALPHA': True}])

        self.nlp.add_pipe(self._merge_doc, name='merge_doc', last=True)
        self.nlp.add_pipe(self._match_doc, name='match_doc', last=True)
        self.nlp.add_pipe(self._postproc_doc, name='postproc_doc', last=True)

    @staticmethod
    def _lowercase(__, doc, i, matches):
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = tok._.transformed_text.lower()

    def _stem_word(self, __, doc, i, matches):
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            if self.tokenizing_parameters['stem'] == 'stem':
                tok._.transformed_text = self.stemmer.stem(tok._.transformed_text)
            elif self.tokenizing_parameters['stem'] == 'lemm':
                tok._.transformed_text = self.stemmer.lemmatize(tok._.transformed_text)

    def _normalize(self, __, doc, i, matches):
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = NORMALIZE_RE.sub(r"\1"*self.tokenizing_parameters['normalize'],
                                                      tok._.transformed_text)

    @staticmethod
    def _extract_domain(__, doc, i, matches):
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            found_urls = URLS_RE.findall(tok.text)
            if found_urls:
                tok._.transformed_text = tldextract.extract(found_urls[0]).domain + '_domain'

    @staticmethod
    def _unwrap_domain(__, doc, i, matches):
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            found_urls = URLS_RE.findall(tok.text)
            if found_urls:
                try:
                    unwrapped_url = requests.get(found_urls[0]).url
                    tok._.transformed_text = tldextract.extract(unwrapped_url).domain + '_domain'
                except requests.exceptions.ConnectionError:
                    warnings.warn('Unable to connect to {}. Replacing URL with original domain'.format(found_urls[0]))
                    tok._.transformed_text = tldextract.extract(found_urls[0]).domain + '_domain'

    def _replace_token(self, __, doc, i, matches):
        match_id, start, end = matches[i]
        span = doc[start:end]
        replacement_token = self._replacements[doc.vocab.strings[match_id]]
        for tok in span:
            tok._.transformed_text = replacement_token

    @staticmethod
    def _remove_token(__, doc, i, matches):
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = ''

    def _split_hashtags(self, __, doc, i, matches):
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            poss = self._infer_spaces(tok.text[1:]).split()
            if poss:
                tok._.transformed_text = poss

    def _hashtag_postprocess(self, __, doc, i, matches):
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            if self.tokenizing_parameters['hashtags']:
                tok._.transformed_text = self.tokenizing_parameters['hashtags']
            elif self.tokenizing_parameters['splithashtags']:
                poss = self._infer_spaces(tok.text[1:]).split()
                if poss:
                    tok._.transformed_text = poss
                else:
                    tok._.transformed_text = tok.text
            else:
                tok._.transformed_text = tok.text

    @staticmethod
    def _decontract(sentence):
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

    def _infer_spaces(self, text):
        """
        Infer the location of spaces
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        def best_match(i):
            """
            Find the best match for the first i characters,
            assuming cost has been built for the first i-1 characters

            Returns:
                match_cost, match_length
        """
            candidates = enumerate(reversed(cost[max(0, i-self.maxword):i]))
            return min((c + self.wordcost.get(text[i-k-1:i], 9e999), k+1) for k, c in candidates)

        # build the cost array
        cost = [0]
        for i in range(1, len(text) + 1):
            cur_cost, k = best_match(i)
            cost.append(cur_cost)

        # backtrack to recover the minimial-cost string
        out = []
        i = len(text)
        while i > 0:
            cur_cost, k = best_match(i)
            assert cur_cost == cost[i]
            out.append(text[i-k:i])
            i -= k

        return " ".join(reversed(out))

    def _preprocess_text(self, text):
        if self.remove_nonunicode:
            try:
                text = text.encode('utf-8').decode('unicode-escape')
                text = ''.join(filter(lambda x: x in string.printable, text)).strip()
            except UnicodeDecodeError:
                pass

        if self.decontract:
            text = self._decontract(text)

        text = re.sub(r'([;,!?\(\)\[\]])', r' \1 ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()

    def _merge_doc(self, doc):
        matches = self.merging_matcher(doc)
        spans = []
        for __, start, end in matches:
            spans.append(doc[start:end])
        for span in spans:
            span.merge()

        for tok in doc:
            tok._.transformed_text = tok.text

        return doc

    def _match_doc(self, doc):
        self.matcher(doc)
        return doc

    def _postproc_doc(self, doc):
        doc._.tokens = []
        for tok in doc:
            if isinstance(tok._.transformed_text, list):
                doc._.tokens.extend(tok._.transformed_text)
            elif tok._.transformed_text.strip() != '':
                if '.' in tok._.transformed_text:
                    if self.tokenizing_parameters['removepunct']:
                        doc._.tokens.extend(tok._.transformed_text.split('.'))
                    else:
                        doc._.tokens.extend(re.split(r'(\W)', tok._.transformed_text))
                else:
                    doc._.tokens.append(tok._.transformed_text.strip())
        return doc

    def tokenize(self, text):
        """
        Tokenize document

        Parameters
        ----------
        text: str
            Document to tokenize
        """
        text = self._preprocess_text(text)
        doc = self.nlp(text)
        return doc._.tokens

    def tokenize_docs(self, texts, batch_size=10000, n_threads=1):
        """
        Tokenize documents in batches

        Parameters
        ----------
        texts: iterable
            Iterable with documents to process
        batch_size: int, default: 10000
            Batch size for processing
        n_threads: int, deafult: 1
            Number of parallel threads
        """
        texts = [self._preprocess_text(text) for text in texts]
        all_tokens = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size, n_threads=n_threads):
            all_tokens.append(doc._.tokens)

        return all_tokens
