from redditscore.SpacyTokenizer import SpacyTokenizer
import re
import pytest

trump_rant ='@realDonaldTrump #fucktrump WHO ELECTED this Guy?! 😭'
doc_emoji = '😍 😭 😩???!!!!'
vova_text = 'Vladimir Putin is the BEST AND AMAZING'
norm_text = 'eeeeeeeeboy this shiiit is good'
quotes_text = '"PBR is the best beer" - said no one ever'
stem_text = 'who has stolen my vodka friends'
punct_text = "this is the text, which contains a lot of punctuation!!! amazing, isn't it? who knows..."
break_text = """I
love
linebreaks"""
nonunicode_text = "no love for russian language пиздец товарищи 😭"
decontract_text = "I've been waiting to drink this beer! I won't give it to you"
hashtag_text = "#makeamericagreatagain #makerussiadrunkagain #MAGA"
replacement_text = "http://fscorelab.ru is number 1 site according to @crazyfrogspb. http://www.google.com"
replacement_text2 = "en919@nyu.edu was hacked by u/AngryConservative from /r/The_Donald #americanhacking"
spartak_text = "Spartak is a champion! Spartalke is the best!"
english_stop = "for on you"
russian_stop = "привет, я Женя"
story_of_my_life = """
	Hi, my name is @crazyfrogspb. I loooove beer. Plato once said - "He was a wise man who invented beer".
	Not a bad way to phrase it. #anotherpintplease
	by the way, don't forget to visit https://github.com/crazyfrogspb/RedditScore.
	I'm also on Reddit as u/crazyfrogspb. I especially love /r/machinelearning
	Sending my love to you. As they say - "stay safe"! ❤ 24
	"""

def test_emoji():
	tokenizer = SpacyTokenizer(pos_emojis=True, neg_emojis=True, neutral_emojis=True)
	tokens = tokenizer.tokenize_doc(doc_emoji)
	assert tokens == ['POS_EMOJI', 'NEG_EMOJI', 'NEG_EMOJI']

def test_repeated():
	tokenizer = SpacyTokenizer(pos_emojis=True, neg_emojis=True, neutral_emojis=True)
	for i in range(100):
		tokens = tokenizer.tokenize_doc(trump_rant)
		
def test_lowercase_keepcaps():
	tokenizer = SpacyTokenizer(lowercase=True, keepcaps=True)
	tokens = tokenizer.tokenize_doc(vova_text)
	assert tokens == ['vladimir', 'putin', 'is', 'the', 'BEST', 'AND', 'AMAZING']
	tokenizer = SpacyTokenizer(lowercase=True, keepcaps=False)
	tokens = tokenizer.tokenize_doc(vova_text)
	assert tokens == ['vladimir', 'putin', 'is', 'the', 'best', 'and', 'amazing']
	tokenizer = SpacyTokenizer(lowercase=False, keepcaps=False)
	tokens = tokenizer.tokenize_doc(vova_text)
	assert tokens == ['Vladimir', 'Putin', 'is', 'the', 'BEST', 'AND', 'AMAZING']
	
def test_normalize():
	tokenizer = SpacyTokenizer(normalize=3)
	tokens = tokenizer.tokenize_doc(norm_text)
	assert tokens == ['eeeboy', 'this', 'shiiit', 'is', 'good']
	tokenizer = SpacyTokenizer(normalize=2)
	tokens = tokenizer.tokenize_doc(norm_text)
	assert tokens == ['eeboy', 'this', 'shiit', 'is', 'good']
	
def test_ignorequotes():
	tokenizer = SpacyTokenizer(ignorequotes=True, removepunct=True)
	tokens = tokenizer.tokenize_doc(quotes_text)
	assert tokens == ['said', 'no', 'one', 'ever']

def test_stop_keep():
	tokenizer = SpacyTokenizer(ignorestopwords=['vladimir', 'putin', 'and'], keepwords=['putin'], lowercase=False)
	tokens = tokenizer.tokenize_doc(vova_text)
	assert tokens == ['Putin', 'is', 'the', 'BEST', 'AMAZING']
	tokenizer = SpacyTokenizer(ignorestopwords='english')
	tokens = tokenizer.tokenize_doc(english_stop)
	assert tokens == []
	tokenizer = SpacyTokenizer(ignorestopwords='russian')
	tokens = tokenizer.tokenize_doc(russian_stop)
	assert tokens == ['привет', 'женя']

def test_stem():
	tokenizer = SpacyTokenizer(stem='stem')
	tokens = tokenizer.tokenize_doc(stem_text)
	assert tokens == ['who', 'ha', 'stolen', 'my', 'vodka', 'friend']

def test_removepunct():
	tokenizer = SpacyTokenizer(removepunct=True)
	tokens = tokenizer.tokenize_doc(punct_text)
	print(tokens)
	assert tokens == ['this', 'is', 'the', 'text', 'which', 'contains', 'a', 'lot', 'of', 'punctuation', 'amazing', "isn't", 'it', 'who', 'knows']

def test_removebreaks():
	tokenizer = SpacyTokenizer(removebreaks=True)
	tokens = tokenizer.tokenize_doc(break_text)
	assert tokens == ['I', 'love', 'linebreaks']

def test_remove_nonunicode():
	tokenizer = SpacyTokenizer(remove_nonunicode=True)
	tokens = tokenizer.tokenize_doc(nonunicode_text)
	assert tokens == ['no', 'love', 'for', 'russian', 'language']

def test_decontract():
	tokenizer = SpacyTokenizer(decontract=True)
	tokens = tokenizer.tokenize_doc(decontract_text)
	assert tokens == ['I', 'have', 'been', 'waiting', 'to', 'drink', 'this', 'beer', 'I', 'will', 'not', 'give', 'it', 'to', 'you']

def test_splithashtags():
	tokenizer = SpacyTokenizer(splithashtags=True, hashtags=False)
	tokens = tokenizer.tokenize_doc(hashtag_text)
	assert tokens == ['make', 'america', 'great', 'again', 'make', 'russia', 'drunk', 'again', 'maga']

def test_replacement():
	tokenizer = SpacyTokenizer(twitter_handles='handle', urls='url', hashtags='hashtag', 
                                    numbers='number', subreddits='subreddit', reddit_usernames='redditor', 
                                    emails='email')
	tokens = tokenizer.tokenize_doc(replacement_text)
	assert tokens == ['url', 'is', 'number', 'number', 'site', 'according', 'to', 'handle', 'url']
	tokens = tokenizer.tokenize_doc(replacement_text2)
	assert tokens == ['email', 'was', 'hacked', 'by', 'redditor', 'from', 'subreddit', 'hashtag']

def test_extra_patterns():
	tokenizer = SpacyTokenizer(extra_patterns=[('zagovor', re.compile(('([S,s]partak|[S,s]paratka|[S,s]partalke)')), 'GAZPROM')])
	tokens = tokenizer.tokenize_doc(spartak_text)
	assert tokens == ['GAZPROM', 'is', 'a', 'champion', 'GAZPROM', 'is', 'the', 'best']

def test_tokenizing():
	tokenizer = SpacyTokenizer(lowercase=True, keepcaps=True, normalize=3, ignorequotes=True, ignorestopwords=['is', 'are', 'am', 'not', 'a', 'the'], 
                                    keepwords=['not'], stem=False, removepunct=True, removebreaks=True, remove_nonunicode=False, decontract=False, 
                                    splithashtags=True, twitter_handles='TOKENTWITTERHANDLE', urls='', hashtags=False, 
                                    numbers=False, subreddits='TOKENSUBREDDIT', reddit_usernames='TOKENREDDITOR', 
                                    emails='TOKENEMAIL', extra_patterns=None, pos_emojis=True, neg_emojis=None, neutral_emojis=None)

	
	tokens = tokenizer.tokenize_doc(story_of_my_life)
	correct_answer = ['hi', 'my', 'name', 'TOKENTWITTERHANDLE', 'I', 'looove', 'beer', 'plato', 'once', 'said', 'not', 'bad', 'way', 'to', 
					  'phrase', 'it', 'another', 'pint', 'please', 'by', 'way', "don't", 'forget', 'to', 'visit', "i'm", 'also', 'on', 
					  'reddit', 'as', 'TOKENREDDITOR', 'I', 'especially', 'love', 'TOKENSUBREDDIT', 'sending', 'my', 'love', 'to', 'you', 
					  'as', 'they', 'say', 'POS_EMOJI', '24']
	assert tokens == correct_answer
