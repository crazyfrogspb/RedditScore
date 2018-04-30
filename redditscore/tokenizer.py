# -*- coding: utf-8 -*-
"""
CrazyTokenizer: spaCy-based tokenizer with Twitter- and Reddit-specific features

Splitting hashtags is based on the idea from
https://stackoverflow.com/questions/11576779/how-to-extract-literal-words-from-a-consecutive-string-efficiently

Author: Evgenii Nikitin <e.nikitin@nyu.edu>

Part of https://github.com/crazyfrogspb/RedditScore project

Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""

import json
import os
import re
import string
import sys
import warnings
from collections import OrderedDict
from http import client
from math import log
from socket import gaierror
from urllib import parse

import requests
import tldextract
from bs4 import BeautifulSoup
from eventlet.green.urllib.request import urlopen
from eventlet.timeout import Timeout
from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.tokens import Doc, Token

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    warnings.warn(
        'nltk could not be imported, some features will be unavailable')

Token.set_extension('transformed_text', default='', force=True)
Doc.set_extension('tokens', default='', force=True)

TIMEOUT = 3.0


POS_EMOJIS = [u'😂', u'❤', u'♥', u'😍', u'😘', u'😊', u'👌', u'💕',
              u'👏', u'😁', u'☺', u'♡', u'👍', u'✌', u'😏', u'😉', u'🙌', u'😄']
NEG_EMOJIS = [u'😭', u'😩', u'😒', u'😔', u'😱']
NEUTRAL_EMOJIS = [u'🙏']

NORMALIZE_RE = re.compile(r"([a-zA-Z])\1\1+")
ALPHA_DIGITS_RE = re.compile(r"[a-zA-Z0-9_]+")
TWITTER_HANDLES_RE = re.compile(r"@\w{1,15}")
REDDITORS_RE = re.compile(r"u/\w{1,20}")
SUBREDDITS_RE = re.compile(r"/r/\w{1,20}")
QUOTES_RE = re.compile(r'^".*"$')
REDDIT_QUOTES_RE = re.compile(r'&gt;[^\n]+\n')
BREAKS_RE = re.compile(r"[\r\n]+")
URLS_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\ ),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

UTF_CHARS = r'a-z0-9_\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff'
TAG_EXP = r'(^|[^0-9A-Z&/]+)(#|\uff03)([0-9A-Z_]*[A-Z_]+[%s]*)' % UTF_CHARS
HASHTAGS_RE = re.compile(TAG_EXP, re.UNICODE | re.IGNORECASE)

URL_SHORTENERS = ['t', 'bit', 'goo', 'tinyurl']

DECONTRACTIONS = OrderedDict([("won't", "will not"), ("can't", "can not"),
                              ("n't", " not"), ("'re", " are"), ("'s", " is"),
                              ("'d", " would"), ("'ll", " will"),
                              ("'t", " not"), ("'ve", " have"),
                              ("'m", " am")])

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         os.path.join('data'))

with open(os.path.join(DATA_PATH, 'emojis_utf.json')) as f:
    EMOJIS_UTF = json.load(f)
with open(os.path.join(DATA_PATH, 'emojis_unicode.json')) as f:
    EMOJIS_UNICODE = json.load(f)
with open(os.path.join(DATA_PATH, 'latin_chars.json')) as f:
    LATIN_CHARS = json.load(f)

EMOJIS_UTF_RE = re.compile(r"\\x", re.IGNORECASE)
EMOJIS_UNICODE_RE = re.compile(r"u\+", re.IGNORECASE)
EMOJIS_UTF_NOSPACE_RE = re.compile(r'(?<!x..)(\\x)', re.IGNORECASE)
EMOJIS_UNICODE_NOSPACE_RE = re.compile(r'(\D{2,})(U\+)', re.IGNORECASE)
LATIN_CHARS_RE = re.compile(r'\\xe2\\', re.IGNORECASE)

EMOJIS_UTF_PATS = {}
for key, value in EMOJIS_UTF.items():
    EMOJIS_UTF_PATS[key] = re.compile(re.escape(key), re.IGNORECASE)
EMOJIS_UNICODE_PATS = {}
for key, value in EMOJIS_UNICODE.items():
    EMOJIS_UNICODE_PATS[key] = re.compile(re.escape(key), re.IGNORECASE)
LATIN_CHARS_PATS = {}
for key, value in LATIN_CHARS.items():
    LATIN_CHARS_PATS[key] = re.compile(re.escape(key), re.IGNORECASE)


def alpha_digits_check(text):
    return bool(ALPHA_DIGITS_RE.fullmatch(text))


def hashtag_check(text):
    return bool(HASHTAGS_RE.fullmatch(text))


def twitter_handle_check(text):
    return bool(TWITTER_HANDLES_RE.fullmatch(text))


def retokenize_check(text):
    if (text.count('@') > 1 or text.count('#') > 1) and text.count(' ') == 0:
        return True
    elif (text.count('@') == 1 or text.count('#') == 1) \
            and text.startswith('@') is False and text.startswith('#') is False:
        return True
    return False


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, length)]


def unshorten_url(url, url_shorteners=None, verbose=False):
    # Fast URL domain extractor
    domain = tldextract.extract(url).domain
    if url_shorteners is not None and domain not in url_shorteners:
        return domain
    parsed = parse.urlparse(url)
    if parsed.scheme == 'http':
        h = client.HTTPConnection(parsed.netloc)
    elif parsed.scheme == 'https':
        h = client.HTTPSConnection(parsed.netloc)
    else:
        return domain
    resource = parsed.path
    if parsed.query != "":
        resource += "?" + parsed.query
    try:
        h.request('HEAD', resource)
    except (TimeoutError, ConnectionRefusedError,
            ConnectionResetError, gaierror):
        if verbose:
            warnings.warn('Connection error for {}'.format(url))
        return domain
    response = h.getresponse()
    if response.status // 100 == 3 and response.getheader('Location'):
        return unshorten_url(response.getheader('Location'),
                             URL_SHORTENERS, verbose)
    else:
        return domain


def get_url_title(url, verbose=False):
    soup = None
    try:
        with Timeout(TIMEOUT, False):
            response = urlopen(url)
            if 'text/html' not in response.getheader('Content-Type'):
                warnings.warn("Url {} is not a text/html page".format(url))
                return ''
            soup = BeautifulSoup(response, "lxml")
    except Exception:
        if verbose:
            warnings.warn("Couldn't extract title from url {}".format(url))
        return ''
    if soup is None or soup.title is None or soup.title.string is None:
        return ''
    return soup.title.string


def get_twitter_realname(twitter_handle):
    try:
        response = requests.get('https://twitter.com/' + twitter_handle)
    except requests.exceptions.ConnectionError:
        warnings.warn(
            "Couldn't extract real name for {}".format(twitter_handle))
        return ''
    soup = BeautifulSoup(response.text, "lxml")
    if soup.title is not None:
        realname = soup.title.text.split('(')[0]
    else:
        realname = ''
    if 'Twitter' in realname:
        return ''
    else:
        return realname


class CrazyTokenizer(object):
    """
    Tokenizer with Reddit- and Twitter-specific options

    Parameters
    ----------
    lowercase : bool, optional
        If True, lowercase all tokens. Defaults to True.

    keepcaps: bool, optional
        If True, keep ALL CAPS WORDS uppercased. Defaults to False.

    normalize: int or bool, optional
        If not False, perform normalization of repeated charachers
        ("awesoooooome" -> "awesooome"). The value of parameter
        determines the number of occurences to keep. Defaults to 3.

    ignore_quotes: bool, optional
        If True, ignore tokens contained within double quotes.
        Defaults to False.

    ignore_reddit_quotes: bool, optional
        If True, remove quotes from the Reddit comments. Defaults to False.

    ignore_stopwords: str, list, or boolean, optional
        Whether to ignore stopwords

        - str: language to get a list of stopwords for from NLTK package
        - list: list of stopwords to remove
        - True: use built-in list of the english stop words
        - False: keep all tokens

        Defaults to False

    stem: {False, 'stem', 'lemm'}, optional
        Whether to perform word stemming

        - False: do not perform word stemming
        - 'stem': use PorterStemmer from NLTK package
        - 'lemm': use WordNetLemmatizer from NLTK package

    remove_punct: bool, optional
        If True, remove punctuation tokens. Defaults to True.

    remove_breaks: bool, optional
        If True, remove linebreak tokens. Defaults to True.

    decontract: bool, optional
        If True, attempt to expand certain contractions. Defaults to False.
        Example: "'ll" -> " will"

    numbers, subreddits, reddit_usernames, emails:
    False or str, optional
        Replacement of the different types of tokens

        - False: leaves these tokens intact
        - str: replacement token
        - '': removes all occurrences of these tokens

    twitter_handles: False, 'realname' or str, optional
        Processing of twitter handles

        - False: do nothing
        - str: replacement token
        - 'realname': replace with the real screen name of Twitter account
        - 'split': split handles using Viterbi algorithm

        Example: "#vladimirputinisthebest" -> "vladimir putin is the best"

    hashtags: False or str, optional
        Processing of hashtags

        - False: do nothing
        - str: replacement token
        - 'split': split hashtags according using Viterbi algorithm

    urls: False or str, optional
        Replacement of parsed URLs

        - False: leave URL intact
        - str: replacement token
        - dict: replace all URLs stored in keys with the corresponding values
        - '': removes all occurrences of these tokens
        - 'domain': extract domain ("http://cnn.com" -> "cnn")
        - 'domain_unwrap_fast': extract domain after unwraping links
        for a list of URL shorteners (goo.gl, t.co, bit.ly, tinyurl.com)
        - 'domain_unwrap': extract domain after unwraping all links
        - 'title': extract and tokenize title of each link after unwraping it

        Defaults to False.

    extra_patterns: None or list of tuples, optional
        Replacement of any user-supplied extra patterns.
        Tuples must have the following form: (name, re_pattern, replacement_token):

        - name (str): name of the pattern
        - re_pattern (_sre.SRE_Pattern): compiled re pattern
        - replacement_token (str): replacement token

        Defaults to None

    keep_untokenized: None or list, optional
        List of expressions to keep untokenized

        Example: ["New York", "Los Angeles", "San Francisco"]

    whitespaces_to_underscores: boolean, optional
        If True, replace all whitespace characters with
        underscores in the final tokens. Defaults to True.

    remove_nonunicode: boolean, optional
        If True, remove all non-unicode characters. Defaults to False.

    pos_emojis, neg_emojis, neutral_emojis: None, True, or list, optional
        Replace positive, negative, and neutral emojis with the special tokens

        - None: do not perform replacement
        - True: perform replacement of the default lists of emojis
        - list: list of emojis to replace

    print_url_warnings: bool, optional
        If True, print URL-related warnings. Defaults to False.

    latin_chars_fix: bool, optional
        Try applying this fix if you have a lot of \\xe2\\x80\\x99-like
        or U+1F601-like strings in your data. Defaults to False.
    """

    def __init__(self, lowercase=True, keepcaps=False, normalize=3,
                 ignore_quotes=False, ignore_reddit_quotes=False,
                 ignore_stopwords=False, stem=False,
                 remove_punct=True, remove_breaks=True, decontract=False,
                 twitter_handles=False, urls=False, hashtags=False,
                 numbers=False, subreddits=False, reddit_usernames=False,
                 emails=False, extra_patterns=None, keep_untokenized=None,
                 whitespaces_to_underscores=True, remove_nonunicode=False,
                 pos_emojis=None, neg_emojis=None, neutral_emojis=None,
                 print_url_warnings=False, latin_chars_fix=False):
        self.params = locals()

        self._nlp = English()
        self._merging_matcher = Matcher(self._nlp.vocab)
        self._matcher = Matcher(self._nlp.vocab)

        self._replacements = {}
        self._domains = {}
        self._realnames = {}
        self._stopwords = None

        alpha_digits_flag = self._nlp.vocab.add_flag(alpha_digits_check)
        hashtag_flag = self._nlp.vocab.add_flag(hashtag_check)
        twitter_handle_flag = self._nlp.vocab.add_flag(twitter_handle_check)

        self._merging_matcher.add(
            'HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])
        self._merging_matcher.add(
            'SUBREDDIT', None,
            [{'ORTH': '/r'}, {'ORTH': '/'}, {alpha_digits_flag: True}],
            [{'ORTH': 'r'}, {'ORTH': '/'}, {alpha_digits_flag: True}])
        self._merging_matcher.add('REDDIT_USERNAME', None,
                                  [{'ORTH': '/u'}, {'ORTH': '/'},
                                      {alpha_digits_flag: True}],
                                  [{'ORTH': 'u'}, {'ORTH': '/'},
                                   {alpha_digits_flag: True}])

        if isinstance(ignore_stopwords, str) and ('nltk' in sys.modules):
            try:
                self._stopwords = stopwords.words(ignore_stopwords)
            except OSError:
                raise ValueError(
                    'Language {} was not found by NLTK'.format(ignore_stopwords))
        elif ignore_stopwords is True:
            self._matcher.add('STOPWORDS', self._remove_token,
                              [{'IS_STOP': True}])
        elif isinstance(ignore_stopwords, list):
            self._stopwords = [word.lower() for word in ignore_stopwords]
        elif ignore_stopwords is not False:
            raise TypeError('Type {} is not supported by ignore_stopwords parameter or NLTK is not installed'.format(
                type(ignore_stopwords)))

        if lowercase and (not keepcaps):
            self._matcher.add('LOWERCASE', self._lowercase,
                              [{'IS_LOWER': False}])
        elif lowercase and keepcaps:
            self._matcher.add('LOWERCASE', self._lowercase, [
                {'IS_LOWER': False, 'IS_UPPER': False}])

        if remove_punct:
            self._matcher.add('PUNCTUATION', self._remove_token, [
                {'IS_PUNCT': True}])

        if remove_breaks:
            def break_check(text):
                return bool(BREAKS_RE.fullmatch(text))
            break_flag = self._nlp.vocab.add_flag(break_check)
            self._matcher.add('BREAK', self._remove_token, [{break_flag: True}])

        if normalize:
            def normalize_check(text):
                return bool(NORMALIZE_RE.search(text))
            normalize_flag = self._nlp.vocab.add_flag(normalize_check)
            self._matcher.add('NORMALIZE', self._normalize,
                              [{normalize_flag: True}])

        if numbers is not False:
            self._matcher.add('NUMBER', self._replace_token,
                              [{'LIKE_NUM': True}])
            self._replacements['NUMBER'] = numbers

        if urls is not False:
            if urls in ['domain', 'domain_unwrap_fast',
                        'domain_unwrap', 'title']:
                self._urls = urls
                self._matcher.add('URL', self._process_url, [
                    {'LIKE_URL': True}])
            elif isinstance(urls, dict):
                self._domains = urls
                self._urls = 'domain_unwrap_fast'
                self._matcher.add('URL', self._process_url, [
                    {'LIKE_URL': True}])
            else:
                self._matcher.add('URL', self._replace_token,
                                  [{'LIKE_URL': True}])
                self._replacements['URL'] = urls

        if emails is not False:
            self._matcher.add('EMAIL', self._replace_token,
                              [{'LIKE_EMAIL': True}])
            self._replacements['EMAIL'] = emails

        if reddit_usernames is not False:
            def reddit_username_check(text):
                return bool(REDDITORS_RE.fullmatch(text))
            reddit_username_flag = self._nlp.vocab.add_flag(
                reddit_username_check)
            self._matcher.add('REDDIT_USERNAME', self._replace_token, [
                {reddit_username_flag: True}])
            self._replacements['REDDIT_USERNAME'] = reddit_usernames

        if subreddits is not False:
            def subreddit_check(text):
                return bool(SUBREDDITS_RE.fullmatch(text))
            subreddit_flag = self._nlp.vocab.add_flag(subreddit_check)
            self._matcher.add('SUBREDDIT', self._replace_token,
                              [{subreddit_flag: True}])
            self._replacements['SUBREDDIT'] = subreddits

        if twitter_handles:
            self._matcher.add('TWITTER_HANDLE', self._handles_postprocess,
                              [{twitter_handle_flag: True}])

        if hashtags:
            self._matcher.add('HASHTAG', self._hashtag_postprocess, [
                {hashtag_flag: True}])

        if hashtags == 'split' or twitter_handles == 'split':
            file = os.path.join(DATA_PATH, 'wordsfreq_wiki2.txt')
            with open(file) as f:
                self._words = f.read().split()
            self._wordcost = dict((k, log((i + 1) * log(len(self._words))))
                                  for i, k in enumerate(self._words))
            self._maxword = max(len(x) for x in self._words)

        if twitter_handles == 'realname':
            with open(os.path.join(DATA_PATH, 'realnames.json')) as f:
                self._realnames = json.load(f)

        if ignore_quotes:
            self._merging_matcher.add('QUOTE', None, [{'ORTH': '"'}, {
                'OP': '*', 'IS_ASCII': True}, {'ORTH': '"'}])

            def doublequote_check(text):
                return bool(QUOTES_RE.fullmatch(text))
            doublequote_flag = self._nlp.vocab.add_flag(doublequote_check)
            self._matcher.add('DOUBLE_QUOTES', self._remove_token, [
                {doublequote_flag: True}])

        if self._stopwords:
            def stopword_check(text):
                return bool(text.lower() in self._stopwords)
            stopword_flag = self._nlp.vocab.add_flag(stopword_check)
            self._matcher.add('STOPWORD', self._remove_token,
                              [{stopword_flag: True}])

        if keep_untokenized is not None:
            if not isinstance(keep_untokenized, list):
                raise ValueError(
                    "keep_untokenized has to be either None or a list")
            for i, phrase in enumerate(keep_untokenized):
                phrase_tokens = phrase.split(' ')
                rule = []
                for token in phrase_tokens:
                    rule.append({'LOWER': token.lower()})
                self._merging_matcher.add('RULE_' + str(i), None, rule)

        if pos_emojis:
            if not isinstance(pos_emojis, list):
                pos_emojis = POS_EMOJIS
            pos_patterns = [[{'ORTH': emoji}] for emoji in pos_emojis]
            self._matcher.add('HAPPY', self._replace_token, *pos_patterns)
            self._replacements['HAPPY'] = 'POS_EMOJI'

        if neg_emojis:
            if not isinstance(neg_emojis, list):
                neg_emojis = NEG_EMOJIS
            neg_patterns = [[{'ORTH': emoji}] for emoji in neg_emojis]
            self._matcher.add('SAD', self._replace_token, *neg_patterns)
            self._replacements['SAD'] = 'NEG_EMOJI'

        if neutral_emojis:
            if not isinstance(neutral_emojis, list):
                neutral_emojis = NEUTRAL_EMOJIS
            neutral_patterns = [[{'ORTH': emoji}] for emoji in neutral_emojis]
            self._matcher.add('NEUTRAL', self._replace_token, *neutral_patterns)
            self._replacements['NEUTRAL'] = 'NEUTRAL_EMOJI'

        if isinstance(extra_patterns, list):
            self._flags = {}
            for name, re_pattern, replacement_token in extra_patterns:
                def flag(text): return bool(re_pattern.match(text))
                self._flags[name] = self._nlp.vocab.add_flag(flag)
                self._matcher.add(name, self._replace_token,
                                  [{self._flags[name]: True}])
                self._replacements[name] = replacement_token

        if stem and ('nltk' in sys.modules):
            if stem == 'stem':
                self._stemmer = PorterStemmer()
            elif stem == 'lemm':
                self._stemmer = WordNetLemmatizer()
            else:
                raise ValueError(
                    'Stemming method {} is not supported'.format(stem))
            self._matcher.add('WORD_TO_STEM', self._stem_word,
                              [{'IS_ALPHA': True}])

        retokenize_flag = self._nlp.vocab.add_flag(retokenize_check)
        self._matcher.add('RETOKENIZE', self._retokenize,
                          [{retokenize_flag: True, 'IS_PUNCT': False,
                            'LIKE_URL': False, 'LIKE_EMAIL': False,
                            'LIKE_NUM': False, hashtag_flag: False,
                            twitter_handle_flag: False}])

        self._nlp.add_pipe(self._merge_doc, name='merge_doc', last=True)
        self._nlp.add_pipe(self._match_doc, name='match_doc', last=True)
        self._nlp.add_pipe(self._postproc_doc, name='postproc_doc', last=True)

    @staticmethod
    def _lowercase(__, doc, i, matches):
        # Lowercase tokens
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = tok._.transformed_text.lower()

    def _stem_word(self, __, doc, i, matches):
        # Stem tokens
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            if self.params['stem'] == 'stem':
                tok._.transformed_text = self._stemmer.stem(
                    tok._.transformed_text)
            elif self.params['stem'] == 'lemm':
                tok._.transformed_text = self._stemmer.lemmatize(
                    tok._.transformed_text)

    def _normalize(self, __, doc, i, matches):
        # Normalize repeating symbols
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = NORMALIZE_RE.sub(r"\1" * self.params['normalize'],
                                                      tok._.transformed_text)

    def _process_url(self, __, doc, i, matches):
        # Process found URLs
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            found_urls = URLS_RE.findall(tok.text)
            if found_urls:
                if found_urls[0] in self._domains:
                    tok._.transformed_text = self._domains[found_urls[0]]
                elif self._urls == 'domain':
                    tok._.transformed_text = tldextract.extract(
                        found_urls[0]).domain
                elif self._urls != 'title':
                    if self._urls == 'domain_unwrap':
                        domain = unshorten_url(
                            found_urls[0], None,
                            self.params['print_url_warnings'])
                    else:
                        domain = unshorten_url(
                            found_urls[0], URL_SHORTENERS,
                            self.params['print_url_warnings'])
                    self._domains[found_urls[0]] = domain
                    tok._.transformed_text = domain
                elif self._urls == 'title':
                    domain = unshorten_url(found_urls[0], URL_SHORTENERS)
                    if domain != 'twitter':
                        title = get_url_title(
                            found_urls[0], self.params['print_url_warnings'])
                        title = self.tokenize(URLS_RE.sub('', title))
                    else:
                        title = ''
                    tok._.transformed_text = title
                    self._domains[found_urls[0]] = title

    def _replace_token(self, __, doc, i, matches):
        # Replace tokens with something else
        match_id, start, end = matches[i]
        span = doc[start:end]
        replacement_token = self._replacements[doc.vocab.strings[match_id]]
        for tok in span:
            tok._.transformed_text = replacement_token

    @staticmethod
    def _remove_token(__, doc, i, matches):
        # Remove tokens
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            tok._.transformed_text = ''

    def _retokenize(self, __, doc, i, matches):
        # Retokenize
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            text = tok.text
            text = re.sub(r'([#@])', r' \1', text)
            text = re.sub(r'\s{2,}', ' ', text).strip()
            tok._.transformed_text = self.tokenize(text)

    def _infer_spaces(self, text):
        # Infer location of spaces in hashtags
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        def best_match(i):
            # Find the best match for the first i characters
            # assuming costs has been built for the first (i-1) characters
            candidates = enumerate(reversed(cost[max(0, i - self._maxword):i]))
            return min((c + self._wordcost.get(text[i - k - 1:i],
                                               9e999), k + 1) for k, c in candidates)

        cost = [0]
        for i in range(1, len(text) + 1):
            cur_cost, k = best_match(i)
            cost.append(cur_cost)

        out = []
        i = len(text)
        while i > 0:
            cur_cost, k = best_match(i)
            assert cur_cost == cost[i]
            out.append(text[i - k:i])
            i -= k

        return list(reversed(out))

    def _handles_postprocess(self, __, doc, i, matches):
        # Process twitter handles
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            if self.params['twitter_handles'] == 'realname':
                if tok.text in self._realnames:
                    tok._.transformed_text = self._realnames[tok.text]
                else:
                    handle = get_twitter_realname(tok.text)
                    realname = self.tokenize(TWITTER_HANDLES_RE.sub('', handle))
                    tok._.transformed_text = realname
                    self._realnames[tok.text] = realname
            elif self.params['twitter_handles'] == 'split':
                poss = self._infer_spaces(tok._.transformed_text[1:])
                if poss:
                    tok._.transformed_text = poss
            else:
                tok._.transformed_text = self.params['twitter_handles']

    def _hashtag_postprocess(self, __, doc, i, matches):
        # Process hashtags
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            if self.params['hashtags'] == 'split':
                poss = self._infer_spaces(tok._.transformed_text[1:])
                if poss:
                    tok._.transformed_text = poss
            else:
                tok._.transformed_text = self.params['hashtags']

    @staticmethod
    def _decontract(text):
        # Expand contractions
        for contraction, decontraction in DECONTRACTIONS.items():
            text = re.sub(contraction, decontraction, text)
        return text

    def _preprocess_text(self, text):
        # Do some preprocessing
        if self.params['remove_nonunicode']:
            try:
                text = text.encode('utf-8').decode('unicode-escape')
                text = ''.join(
                    filter(lambda x: x in string.printable, text)).strip()
            except UnicodeDecodeError:
                warnings.warn(
                    '(UnicodeDecodeError while trying to remove non-unicode characters')
        if self.params['decontract']:
            text = self._decontract(text)

        if self.params['latin_chars_fix']:
            if EMOJIS_UTF_RE.findall(text):
                text = EMOJIS_UTF_NOSPACE_RE.sub(r' \1', text)
                for utf_code, emoji in EMOJIS_UTF.items():
                    text = EMOJIS_UTF_PATS[utf_code].sub(emoji, text)

            if EMOJIS_UNICODE_RE.findall(text):
                text = EMOJIS_UNICODE_NOSPACE_RE.sub(r'\1 \2', text)
                for utf_code, emoji in EMOJIS_UNICODE.items():
                    text = EMOJIS_UNICODE_PATS[utf_code].sub(emoji, text)

            if LATIN_CHARS_RE.findall(text):
                for _hex, _char in LATIN_CHARS.items():
                    text = LATIN_CHARS_PATS[_hex].sub(_char, text)

        if self.params['ignore_reddit_quotes']:
            text = REDDIT_QUOTES_RE.sub(text, ' ')

        text = text.replace('.@', '. @')
        text = re.sub(r'([*;,!?\(\)\[\]])', r' \1', text)
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()

    def _merge_doc(self, doc):
        # Perform merging for certain types of tokens
        matches = self._merging_matcher(doc)
        spans = []
        for __, start, end in matches:
            spans.append(doc[start:end])
        for span in spans:
            span.merge()
        for tok in doc:
            tok._.transformed_text = tok.text

        return doc

    def _match_doc(self, doc):
        # Perform all additional processing
        self._matcher(doc)
        return doc

    def _postproc_doc(self, doc):
        # Perform postprocessing
        doc._.tokens = []
        for tok in doc:
            if isinstance(tok._.transformed_text, list):
                doc._.tokens.extend(tok._.transformed_text)
            elif tok._.transformed_text.strip() != '':
                if self.params['whitespaces_to_underscores']:
                    tok._.transformed_text = "_".join(
                        tok._.transformed_text.split())
                doc._.tokens.append(tok._.transformed_text.strip())
        return doc

    def tokenize(self, text):
        """
        Tokenize document

        Parameters
        ----------
        text : str
            Document to tokenize

        Returns
        -------
        list
            List of tokens

        Examples
        --------
        >>> from redditscore.tokenizer import CrazyTokenizer
        >>> tokenizer = CrazyTokenizer(splithashtags=True, hashtags=False)
        >>> tokenizer.tokenize("#makeamericagreatagain")
        ["make", "america", "great", "again"]
        """
        if not isinstance(text, str):
            warnings.warn('Document {} is not a string'.format(text))
            return []
        text = self._preprocess_text(text)
        doc = self._nlp(text)
        return doc._.tokens
