# RedditScore
Package for performing Reddit-based text analysis

Includes:
- SpaCy-based tokenizer with Twitter- and Reddit-specific features
- Tools for easy training of Reddit-based models and performing analysis
- Tools to use pre-trained models

Example:

	from redditscore.tokenizer import CrazyTokenizer
	trump_rant = "@realDonaldTrump #fucktrump WHO ELECTED this Guy?! 😭'"
	tokenizer = CrazyTokenizer(removepunct=True, ignorequotes=True, ignorestopwords=True, splithashtags=True, neg_emojis=True, hashtags=False)
	tokenizer.tokenize_doc(trump_rant)

	Output:
	['TOKENTWITTERHANDLE', 'fuck', 'trump', 'WHO', 'ELECTED', 'guy', 'NEG_EMOJI']

Comparison with other popular tokenizers:

    reddit = ("[A month ago](https://www.reddit.com/r/WikiLeaks/comments/6cttkj/i_started_mapping_the_cctv_cameras_near_the_seth/)"
          "I mapped out a few of the CCTV cameras near the murder.[Lots of cameras for Seth to walk by,](http://i.imgur.com/P6IeYdB.png)"
          "but never any footage released.Don't worry though.[The DNC got a commemorative bike rack for him.]"
          "(https://www.reddit.com/r/WikiLeaks/comments/6luow9/on_the_one_year_anniversary_of_his_murder_the_dnc/)")

    twitter = "@realDonaldTrump WHO ELECTED this Guy?! #fucktrump https://goo.gl/mUTaKX"

    Reddit tokenization:
    SpaCy Tokenizer: ['[', 'A', 'month', 'ago](https://www.reddit.com', '/', 'r', '/', 'WikiLeaks', '/', 'comments/6cttkj', '/', 'i_started_mapping_the_cctv_cameras_near_the_seth/)I', 'mapped', 'out', 'a', 'few', 'of', 'the', 'CCTV', 'cameras', 'near', 'the', 'murder.[Lots', 'of', 'cameras', 'for', 'Seth', 'to', 'walk', 'by,](http://i.imgur.com', '/', 'P6IeYdB.png)but', 'never', 'any', 'footage', 'released', '.', "Don't", 'worry', 'though.[The', 'DNC', 'got', 'a', 'commemorative', 'bike', 'rack', 'for', 'him.](https://www.reddit.com', '/', 'r', '/', 'WikiLeaks', '/', 'comments/6luow9/on_the_one_year_anniversary_of_his_murder_the_dnc/', ')'],
    NLTK TweetTokenizer: ['[', 'A', 'month', 'ago', ']', '(', 'https://www.reddit.com/r/WikiLeaks/comments/6cttkj/i_started_mapping_the_cctv_cameras_near_the_seth/', ')', 'I', 'mapped', 'out', 'a', 'few', 'of', 'the', 'CCTV', 'cameras', 'near', 'the', 'murder', '.', '[', 'Lots', 'of', 'cameras', 'for', 'Seth', 'to', 'walk', 'by', ',', ']', '(', 'http://i.imgur.com/P6IeYdB.png', ')', 'but', 'never', 'any', 'footage', 'released.Don', "'", 't', 'worry', 'though', '.', '[', 'The', 'DNC', 'got', 'a', 'commemorative', 'bike', 'rack', 'for', 'him', '.', ']', '(', 'https://www.reddit.com/r/WikiLeaks/comments/6luow9/on_the_one_year_anniversary_of_his_murder_the_dnc/', ')'],
    CrazyTokenizer: ['A', 'month', 'ago', 'reddit_domain', 'I', 'mapped', 'out', 'a', 'few', 'of', 'the', 'CCTV', 'cameras', 'near', 'the', 'murder', 'lots', 'of', 'cameras', 'for', 'seth', 'to', 'walk', 'by', 'imgur_domain', 'but', 'never', 'any', 'footage', 'released', "don't", 'worry', 'though', 'the', 'DNC', 'got', 'a', 'commemorative', 'bike', 'rack', 'for', 'him', 'reddit_domain']

    Twitter tokenization:
    SpaCy Tokenizer: ['@realDonaldTrump', 'WHO', 'ELECTED', 'this', 'Guy', '?', '!', '#', 'fucktrump', 'https://goo.gl/mUTaKX'],
    NLTK TweetTokenizer: ['@realDonaldTrump', 'WHO', 'ELECTED', 'this', 'Guy', '?', '!', '#fucktrump', 'https://goo.gl/mUTaKX'],
    CrazyTokenizer: ['TOKENTWITTERHANDLE', 'WHO', 'ELECTED', 'this', 'guy', 'fuck', 'trump', 'cnn_domain']
	
To install package:

	pip install git+https://github.com/crazyfrogspb/RedditScore.git

To cite:

    { 
      @misc{Nikitin2018,
      author = {Nikitin, E.},
      title = {RedditScore},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/crazyfrogspb/RedditScore}},
      commit = {faf15eaed7cb334dfd7213195bbbb68861767d6a}
    }