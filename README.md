# RedditScore
Package for performing Reddit-based text analysis

Includes:
- SpaCy-based tokenizer with Twitter- and Reddit-specific features
- Tools for easy training of Reddit-based models and performing analysis
- Tools to use pre-trained models

Example:

	from redditscore.SpacyTokenizer import SpacyTokenizer
	trump_rant = "@realDonaldTrump #fucktrump WHO ELECTED this Guy?! 😭'"
	tokenizer = SpacyTokenizer(removepunct=True, ignorequotes=True, ignorestopwords=True, splithashtags=True, neg_emojis=True, hashtags=False)
	tokenizer.tokenize_doc(trump_rant)

	Output:
	['TOKENTWITTERHANDLE', 'fuck', 'trump', 'WHO', 'ELECTED', 'guy', 'NEG_EMOJI']
	
To run rests execute the following command from the root directory:
	
	py.test -vv
	
To install package:

	pip install git+https://github.com/crazyfrogspb/RedditScore.git