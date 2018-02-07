import re
import string
from collections import OrderedDict
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer
except ModuleNotFoundError:
    print('nltk could not be imported, some features will be unavilable')
import sys

class TokenReplacer(object):
    def __init__(self, pattern, token):
        self.reg = re.compile(pattern)
        if token in ['', 'REMOVE']:
            self.token = ''
        else:
            self.token = token
    def replace(self, s):
        return self.reg.sub(self.token, s)

class Tokenizer(object):
    twitter_handles_re = r"@\w{1,15}"
    reddit_usernames_re = r"u/\w{1,20}"
    subreddits_re = r"/r/\w{1,20}"
    urls_re = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    hashtags_re = r"#\w+[\w'-]*\w+"
    numbers_re = r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?"
    punct_re = re.compile('[%s]' % re.escape(string.punctuation))
    patterns = ['twitter_handles', 'subreddits', 'reddit_usernames', 'hashtags', 'urls', 'numbers']
    
    def __init__(self, **kwargs):
        self._default_values = dict(lowercase=True, keepcaps=True, normalize=3, ignorequotes=False, ignorestopwords=False, keepwords=None, stem=False, 
                                    removepunct=True, remove_nonunicode=False, decontract=False, splithashtags=True, twitter_handles='TOKENTWITTERHANDLE', 
                                    urls='TOKENURL', hashtags=False, numbers='TOKENNUMBER', subreddits='TOKENSUBREDDIT', reddit_usernames='TOKENREDDITOR', 
                                    extra_patterns=None)
        for (prop, default) in self._default_values.items():
            setattr(self, prop, kwargs.get(prop, default))
            
        self.replacers = []
        for pattern in self.patterns:
            token = getattr(self, pattern)
            if token:
                replacer = TokenReplacer(getattr(self, pattern + '_re'), token)
                self.replacers.append(replacer)
                
        if self.extra_patterns:
            for extra_pattern in extra_patterns:
                replacer = TokenReplacer(extra_pattern[0], extra_pattern[1])
                self.replacers.append(replacer)
                
        if (self.ignorestopwords == True) & ('nltk.stopwords' in sys.modules):
            self._stopwords = stopwords.words('english')
        elif self.ignorestopwords == False:
            self._stopwords = []
        else:
            if self.ignorestopwords is not list:
                raise TypeError("Stop words argument should be a list")
            self._stopwords = self.ignorestopwords
            
              
        if self.keepwords is None:
            self.keepwords=[]
            
    def update(self, **kwargs):
        for keyword in self._default_args:
            if keyword in kwargs:
                setattr(self, keyword, kwargs[keyword])
            
    def tokenize_sentence(self, sentence):
        if not isinstance(sentence, str):
            raise TypeError('Can only tokenize a string, provided type is {}'.format(repr(type(sentence).__name__)))
        for replacer in self.replacers:
            sentence = replacer.replace(sentence)
        if self.decontract:
            sentence = self._decontract(sentence)
        if self.remove_nonunicode:
            try:
                sentence = sentence.encode('utf-8').decode('unicode-escape')
                sentence = ''.join(filter(lambda x: x in string.printable, sentence))
            except UnicodeDecodeError:
                pass
            
        if self._stopwords:
            sentence = [word for word in sentence if word not in self._stopwords]
        
        if 'nltk.stem' in sys.modules:
            if self.stem == 'lemm':
                wnl = WordNetLemmatizer()
                sentence = [wnl.lemmatize(word) for word in sentence]
            elif self.stem == 'stem':
                stemmer=PorterStemmer()
                sentence = [stemmer.stem(word) for word in sentence]
        if self.removepunct:
            sentence = self.punct_re.sub('', sentence)
        return sentence
    
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