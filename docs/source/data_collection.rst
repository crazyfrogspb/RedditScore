Data collection
=========================================

RedditScore library can be used on any kind of text data, but it was
developed specifically for building Reddit-based models any applying them to
Twitter data. This is why RedditScore also includes tools to easily pull
data from these sources.

Reddit Data
--------------------
The easiest way to obtain Reddit data is to use Google BigQuery
`public dataset of Reddit comments <https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.2017_12>`__.
Google BigQuery is not free, however, you can get 365 days trial and 300$ credit
when you register for the first time.

After setting up your account, create a project and copy its Project ID. Next,
create a service account as described
`here <https://cloud.google.com/bigquery/docs/reference/libraries#setting-up-authentitication>`__.
Download JSON file with your private key.

After that, you're good to go! ``get_comments`` function in RedditScore library
will help you to easily collect some Reddit comments. Let's say you want to collect
all comments posted in subreddits /r/BeardAdvice and /r/beards from
Jan 2016 to May 2016. This is how you do it:

>>> from redditscore import get_reddit_data as grd
>>> project_id = "my_first_project" # insert your Project ID here
>>> private_key = 'my_key.json' # insert path to your key file here
>>> subreddits = ['BeardAdvice', 'beards']
>>> df = grd.get_comments(subreddits, ('2016_01', '2016_05'),
>>>                       project_id, private_key, verbose=True)

``df`` is a pandas DataFrame with collected comments.

``get_comments`` function also has a few additional options:

  - Saving results by month into CSV files instead of returning a DataFrame:

  >>> grd.get_comments(subreddits, ('2016_01', '2016_05'), project_id, private_key,
  >>>                  verbose=True, csv_directory='reddit_data')

  - Getting a random sample of comments from each subreddit per month:

  >>> df = grd.get_comments(subreddits, ('2016_01', '2016_05'), project_id, private_key,
  >>>                       verbose=True, comments_per_month=1000)

  - Getting top-scoring comments from each subreddit per month:

  >>> df = grd.get_comments(subreddits, ('2016_01', '2016_05'), project_id, private_key,
  >>>                       verbose=True, comments_per_month=1000, top_scores=True)

  - Filtering out comments with low scores:

  >>> df = grd.get_comments(subreddits, ('2016_01', '2016_05'), project_id, private_key,
  >>>                       verbose=True, score_limit=3)

Twitter Data
--------------------
RedditScore can help you to pull tweets from specific Twitter accounts. First,
you need to create your own Twitter application and receive credentials.
You can do it in `Application Management <https://apps.twitter.com/>`__. Save
your consumer_key, consumer_secret, access_key, and access_secret to JSON file,
and you're ready to collect the data!

>>> import json
>>> from redditscore import get_twitter_data as gtd
>>> cred_path = 'twitter_creds.json'
>>> with open(cred_path) as f:
>>>   twitter_creds = json.load(f)
>>> df = grab_tweets('crazyfrogspb' twitter_creds)

There are a few optional arguments that ``grab_tweets`` function takes:

   - ``timeout``: time in seconds to wait between requests. Increase if you experience issues with Twitter API limits
   - ``fields``: additional fields to pull from tweets (e.g., favorite_count)
   - ``get_more``: if True, use Selenium library to pull tweets beyond infamous 3200 limit. Please note that it is quite slow and you need to have selenium package, browser (Chrome, Firefox, or Safari), and its webdriver installed.
   - ``browser``: which browser to use ('Chrome', 'Firefox', or 'Safari')
   - ``start_date``: date to start pulling additional tweets from (must be in datetime.date format)
