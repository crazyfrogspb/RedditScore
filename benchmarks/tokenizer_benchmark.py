from redditscore.tokenizer import CrazyTokenizer
from spacy.lang.en import English
from nltk.tokenize import TweetTokenizer

nlp = English()
crazy_tokenizer = CrazyTokenizer()
nltk_tokenizer = TweetTokenizer()

reddit = ("[A month ago](https://www.reddit.com/r/WikiLeaks/comments/6cttkj/i_started_mapping_the_cctv_cameras_near_the_seth/)"
          "I mapped out a few of the CCTV cameras near the murder.[Lots of cameras for Seth to walk by,](http://i.imgur.com/P6IeYdB.png)"
          "but never any footage released.Don't worry though.[The DNC got a commemorative bike rack for him.]"
          "(https://www.reddit.com/r/WikiLeaks/comments/6luow9/on_the_one_year_anniversary_of_his_murder_the_dnc/)")

twitter = "@realDonaldTrump WHO ELECTED this Guy?! #fucktrump https://goo.gl/mUTaKX"


def spacy_tokenize(text):
    doc = nlp(text)
    return [tok.text for tok in doc]


reddit_spacy_tokens = spacy_tokenize(reddit)
reddit_nltk_tokens = nltk_tokenizer.tokenize(reddit)
reddit_crazy_tokens = crazy_tokenizer.tokenize(reddit)

twitter_spacy_tokens = spacy_tokenize(twitter)
twitter_nltk_tokens = nltk_tokenizer.tokenize(twitter)
twitter_crazy_tokens = crazy_tokenizer.tokenize(twitter)

print('Reddit tokenization: {linebreak}SpaCy: {spacy_tokens}, {linebreak}NLTK TweetTokenizer: {nltk_tokens}, {linebreak}CrazyTokenizer: {crazy_tokens}'.format(
    spacy_tokens=reddit_spacy_tokens, nltk_tokens=reddit_nltk_tokens, crazy_tokens=reddit_crazy_tokens, linebreak="\n"))
print('Twitter tokenization: {linebreak}SpaCy: {spacy_tokens}, {linebreak}NLTK TweetTokenizer: {nltk_tokens}, {linebreak}CrazyTokenizer: {crazy_tokens}'.format(
    spacy_tokens=twitter_spacy_tokens, nltk_tokens=twitter_nltk_tokens, crazy_tokens=twitter_crazy_tokens, linebreak="\n"))
