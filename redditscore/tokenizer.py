# -*- coding: utf-8 -*-
"""
CrazyTokenizer: SpaCy-based tokenizer with Twitter- and Reddit-specific features

Author: Evgenii Nikitin <e.nikitin@nyu.edu>

Part of https://github.com/crazyfrogspb/RedditScore project

Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""

import os
import re
import string
import sys
import warnings
from http import client
from math import log
from urllib import parse

import requests
import tldextract
from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc, Token

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    warnings.warn(
        'nltk could not be imported, some features will be unavailable')

Token.set_extension('transformed_text', default='', force=True)
Doc.set_extension('tokens', default='', force=True)

POS_EMOJIS = [u'😂', u'❤', u'♥', u'😍', u'😘', u'😊', u'👌', u'💕',
              u'👏', u'😁', u'☺', u'♡', u'👍', u'✌', u'😏', u'😉', u'🙌', u'😄']
NEG_EMOJIS = [u'😭', u'😩', u'😒', u'😔', u'😱']
NEUTRAL_EMOJIS = [u'🙏']

NORMALIZE_RE = re.compile(r"([a-zA-Z])\1\1+")
URLS_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

PREFIX_RE = re.compile(r'''^[\[\("'\s]''')
SUFFIX_RE = re.compile(r'''[\]\)"'?!.\s,:;-]$''')

PREFIX_RE = re.compile(r'''^[\[\("'?!.,:;*\-]''')
SUFFIX_RE = re.compile(r'''[ \] \)"'?!.,:;*\-]$''')
INFIX_RE = re.compile(r'''([.]{3,}|[\]\)\[\("?!,;])''')

URL_SHORTENERS = ['t', 'bit', 'goo', 'tinyurl']


def custom_tokenizer(nlp):
    # Customize SpaCy tokenizer
    return Tokenizer(nlp.vocab, prefix_search=PREFIX_RE.search,
                     suffix_search=SUFFIX_RE.search,
                     infix_finditer=INFIX_RE.finditer)


def unshorten_url(url, url_shorteners):
    # Fast URL domain extractor
    domain = tldextract.extract(url).domain
    if domain not in url_shorteners:
        return domain
    parsed = parse.urlparse(url)
    if parsed.scheme == 'http':
        h = client.HTTPConnection(parsed.netloc)
    elif parsed.scheme == 'https':
        h = client.HTTPSConnection(parsed.netloc)
    else:
        return 'TOKENURL'
    resource = parsed.path
    if parsed.query != "":
        resource += "?" + parsed.query
    h.request('HEAD', resource)
    response = h.getresponse()
    if response.status // 100 == 3 and response.getheader('Location'):
        return unshorten_url(response.getheader('Location'), URL_SHORTENERS)
    elif response.status == 404:
        return 'TOKENURL'
    else:
        return tldextract.extract(url).domain


class CrazyTokenizer(object):
    """
    Tokenizer with Reddit- and Twitter-specific options

    Parameters
    ----------
    lowercase : bool, optional
        If True, lowercase all tokens. Defaults to True.

    keepcaps: bool, optional
        If True, keep ALL CAPS WORDS uppercased. Defaults to True.

    normalize: int or bool, optional
        If not False, perform normalization of repeated charachers
        ("awesoooooome" -> "awesooome"). The value of parameter
        determines the number of occurences to keep. Defaults to 3.

    ignorequotes: bool, optional
        If True, ignore tokens contained within double quotes.
        Defaults to False.

    ignorestopwords: str, list, or False, optional
        Whether to ignore stopwords

        - str: language to get a list of stopwords for from NLTK package
        - list: list of stopwords to remove
        - False: keep all tokens

        Defaults to False

    stem: {False, 'stem', 'lemm'}, optional
        Whether to perform word stemming

        - False: do not perform word stemming
        - 'stem': use PorterStemmer from NLTK package
        - 'lemm': use WordNetLemmatizer from NLTK package

    removepunct: bool, optional
        If True, remove punctuation tokens. Defaults to True.

    removebreaks: bool, optional
        If True, remove linebreak tokens. Defaults to True.

    decontract: bool, optional
        If True, attempt to expand certain contractions. Defaults to False.
        Example: "'ll" -> " will"

    splithashtags: bool, optional
        If True, split hashtags according to word frequency.
        Example: "#vladimirputinisthebest" -> "vladimir putin is the best"
        Defaults to False.

    twitter_handles, hashtags, numbers, subreddits, reddit_usernames, emails:
    False or str, optional
        Replacement of the different types of tokens

        - False: leaves these tokens intact
        - str: replacement token
        - '': removes all occurrences of these tokens

    urls: False or str, optional
        Replacement of parsed URLs

        - False: leave URL intact
        - str: replacement token
        - '': removes all occurrences of these tokens
        - 'domain': extract domain ("http://cnn.com" -> "cnn_domain")
        - 'domain_unwrap_fast': extract domain after unwraping links
            for a list of URL shorteners (goo.gl, t.co, bit.ly, tinyurl.com)
        - 'domain_unwrap': extract domain after unwraping all links

        Defaults to 'domain'.

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
    """

    def __init__(self, lowercase=True, keepcaps=True, normalize=3,
                 ignorequotes=False, ignorestopwords=False, stem=False,
                 removepunct=True, removebreaks=True, decontract=False,
                 splithashtags=False, twitter_handles='TOKENTWITTERHANDLE',
                 urls='domain', hashtags=False, numbers='TOKENNUMBER',
                 subreddits='TOKENSUBREDDIT', reddit_usernames='TOKENREDDITOR',
                 emails='TOKENEMAIL', extra_patterns=None, keep_untokenized=None,
                 whitespaces_to_underscores=True, remove_nonunicode=False,
                 pos_emojis=None, neg_emojis=None, neutral_emojis=None):
        self._nlp = English()
        self._nlp.tokenizer = custom_tokenizer(self._nlp)
        self._merging_matcher = Matcher(self._nlp.vocab)
        self._matcher = Matcher(self._nlp.vocab)
        self._replacements = {}
        self._domains = {}
        self.params = locals()

        self._merging_matcher.add(
            'HASHTAG', None, [{'ORTH': '#'}, {'IS_ASCII': True}])
        self._merging_matcher.add(
            'SUBREDDIT', None, [{'ORTH': '/r'}, {'ORTH': '/'}, {'IS_ASCII': True}])
        self._merging_matcher.add('REDDIT_USERNAME', None, [
            {'ORTH': 'u'}, {'ORTH': '/'}, {'IS_ASCII': True}])

        if isinstance(ignorestopwords, str) and ('nltk' in sys.modules):
            try:
                self._stopwords = stopwords.words(ignorestopwords)
            except OSError:
                raise ValueError(
                    'Language {} was not found by NLTK'.format(ignorestopwords))
        elif isinstance(ignorestopwords, list):
            self._stopwords = [word.lower() for word in ignorestopwords]
        elif (not ignorestopwords) or (ignorestopwords is None):
            self._stopwords = []
        else:
            raise TypeError('Type {} is not supported by ignorestopwords parameter'.format(
                type(ignorestopwords)))

        if lowercase and (not keepcaps):
            self._matcher.add('LOWERCASE', self._lowercase,
                              [{'IS_LOWER': False}])
        elif lowercase and keepcaps:
            self._matcher.add('LOWERCASE', self._lowercase, [
                {'IS_LOWER': False, 'IS_UPPER': False}])

        if removepunct:
            self._matcher.add('PUNCTUATION', self._remove_token, [
                {'IS_PUNCT': True}])

        if removebreaks:
            def break_check(text): return bool(
                re.compile(r"[\r\n]+").fullmatch(text))
            break_flag = self._nlp.vocab.add_flag(break_check)
            self._matcher.add('BREAK', self._remove_token, [{break_flag: True}])

        if normalize:
            def normalize_check(text): return bool(NORMALIZE_RE.search(text))
            normalize_flag = self._nlp.vocab.add_flag(normalize_check)
            self._matcher.add('NORMALIZE', self._normalize,
                              [{normalize_flag: True}])

        if numbers is not False:
            def number_check(text): return bool(re.compile(
                r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?").fullmatch(text))
            number_flag = self._nlp.vocab.add_flag(number_check)
            self._matcher.add('NUMBER', self._replace_token,
                              [{number_flag: True}])
            self._replacements['NUMBER'] = numbers

        if urls is not False:
            if urls == 'domain':
                self._matcher.add('URL', self._extract_domain,
                                  [{'LIKE_URL': True}])
            elif urls == 'domain_unwrap_fast':
                self._matcher.add('URL', self._unwrap_domain_fast, [
                    {'LIKE_URL': True}])
            elif urls == 'domain_unwrap':
                self._matcher.add('URL', self._unwrap_domain,
                                  [{'LIKE_URL': True}])
            else:
                self._matcher.add('URL', self._replace_token,
                                  [{'LIKE_URL': True}])
                self._replacements['URL'] = urls

        if twitter_handles is not False:
            def twitter_handle_check(text): return bool(
                re.compile(r"@\w{1,15}").fullmatch(text))
            twitter_handle_flag = self._nlp.vocab.add_flag(twitter_handle_check)
            self._matcher.add('TWITTER_HANDLE', self._replace_token, [
                {twitter_handle_flag: True}])
            self._replacements['TWITTER_HANDLE'] = twitter_handles

        if reddit_usernames is not False:
            def reddit_username_check(text): return bool(
                re.compile(r"u/\w{1,20}").fullmatch(text))
            reddit_username_flag = self._nlp.vocab.add_flag(
                reddit_username_check)
            self._matcher.add('REDDIT_USERNAME', self._replace_token, [
                {reddit_username_flag: True}])
            self._replacements['REDDIT_USERNAME'] = reddit_usernames

        if subreddits is not False:
            def subreddit_check(text): return bool(
                re.compile(r"/r/\w{1,20}").fullmatch(text))
            subreddit_flag = self._nlp.vocab.add_flag(subreddit_check)
            self._matcher.add('SUBREDDIT', self._replace_token,
                              [{subreddit_flag: True}])
            self._replacements['SUBREDDIT'] = subreddits

        if (hashtags is not False) or splithashtags:
            def hashtag_check(text): return bool(
                re.compile(r"#\w+[\w'-]*\w+").fullmatch(text))
            hashtag_flag = self._nlp.vocab.add_flag(hashtag_check)
            self._matcher.add('HASHTAG', self._hashtag_postprocess, [
                {hashtag_flag: True}])
            if splithashtags:
                file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    os.path.join('data', 'wordsfreq.txt'))
                self._words = open(file).read().split()
                self._wordcost = dict((k, log((i + 1) * log(len(self._words))))
                                      for i, k in enumerate(self._words))
                self._maxword = max(len(x) for x in self._words)

        if emails is not False:
            self._matcher.add('EMAIL', self._replace_token,
                              [{'LIKE_EMAIL': True}])
            self._replacements['EMAIL'] = emails

        if ignorequotes:
            self._merging_matcher.add('QUOTE', None, [{'ORTH': '"'}, {
                'OP': '*', 'IS_ASCII': True}, {'ORTH': '"'}])

            def doublequote_check(text): return bool(
                re.compile(r'^".*"$').fullmatch(text))
            doublequote_flag = self._nlp.vocab.add_flag(doublequote_check)
            self._matcher.add('DOUBLE_QUOTES', self._remove_token, [
                {doublequote_flag: True}])

        if self._stopwords:
            def stopword_check(text): return bool(
                text.lower() in self._stopwords)
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

    @staticmethod
    def _extract_domain(__, doc, i, matches):
        # Extract domain without unwrapping
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            found_urls = URLS_RE.findall(tok.text)
            if found_urls:
                tok._.transformed_text = tldextract.extract(
                    found_urls[0]).domain + '_domain'

    def _unwrap_domain_fast(self, __, doc, i, matches):
        # Extract domain after unwrapping specific URL shortners
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            found_urls = URLS_RE.findall(tok.text)
            if found_urls:
                if found_urls[0] in self._domains:
                    tok._.transformed_text = self._domains[found_urls[0]] + '_domain'
                else:
                    domain = unshorten_url(found_urls[0], URL_SHORTENERS)
                    self._domains[found_urls[0]] = domain
                    tok._.transformed_text = domain + '_domain'

    def _unwrap_domain(self, __, doc, i, matches):
        # Extract domain after unwrapping all URLs
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            found_urls = URLS_RE.findall(tok.text)
            if found_urls:
                if found_urls[0] in self._domains:
                    tok._.transformed_text = self._domains[found_urls[0]] + '_domain'
                else:
                    try:
                        unwrapped_url = requests.head(
                            found_urls[0], allow_redirects=True).url
                        domain = tldextract.extract(unwrapped_url).domain
                        self._domains[found_urls[0]] = domain
                        tok._.transformed_text = domain + '_domain'
                    except requests.exceptions.ConnectionError:
                        warnings.warn(
                            'Unable to connect to {}. Replacing URL with original domain'.format(found_urls[0]))
                        tok._.transformed_text = tldextract.extract(
                            found_urls[0]).domain + '_domain'

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

        return " ".join(reversed(out))

    def _split_hashtags(self, __, doc, i, matches):
        # Split hashtags
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            poss = self._infer_spaces(tok.text[1:]).split()
            if poss:
                tok._.transformed_text = poss

    def _hashtag_postprocess(self, __, doc, i, matches):
        # Process hashtags
        __, start, end = matches[i]
        span = doc[start:end]
        for tok in span:
            if self.params['hashtags']:
                tok._.transformed_text = self.params['hashtags']
            elif self.params['splithashtags']:
                poss = self._infer_spaces(tok.text[1:]).split()
                if poss:
                    tok._.transformed_text = poss
                else:
                    tok._.transformed_text = tok.text
            else:
                tok._.transformed_text = tok.text

    @staticmethod
    def _decontract(sentence):
        # Expand contractions
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

    def _preprocess_text(self, text):
        # Do some preprocessing
        if self.params['remove_nonunicode']:
            try:
                text = text.encode('utf-8').decode('unicode-escape')
                text = ''.join(
                    filter(lambda x: x in string.printable, text)).strip()
            except UnicodeDecodeError:
                warnings.warn(
                    'UnicodeDecodeError while trying to remove non-unicode charachers')
        if self.params['decontract']:
            text = self._decontract(text)

        text = re.sub(r'([;,!?\(\)\[\]])', r' \1 ', text)
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
        text = self._preprocess_text(text)
        doc = self._nlp(text)
        return doc._.tokens

    def tokenize_docs(self, texts, batch_size=10000, n_threads=2):
        """
        Tokenize documents in batches

        Parameters
        ----------
        texts: iterable
            Iterable with documents to process
        batch_size: int, optional
            Batch size for processing
        n_threads: int, optional
            Number of parallel threads

        Returns
        -------
        all_tokens: list of lists
            List with tokenized documents
        """
        texts = [self._preprocess_text(text) for text in texts]
        all_tokens = []
        for doc in self._nlp.pipe(texts, batch_size=batch_size, n_threads=n_threads):
            all_tokens.append(doc._.tokens)
        return all_tokens
