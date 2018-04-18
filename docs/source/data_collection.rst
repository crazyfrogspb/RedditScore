Data collection
=========================================

RedditScore library can be used on any kind of text data, but it was
developed specifically for building Reddit-based models. The easiest way
to obtain Reddit data is to use Google BigQuery
`public dataset of Reddit comments <https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.2017_12>`__.
Google BigQuery is not free, however, you get 365 days trial and 300$ credit
when you register for the first time.

After setting up your account, create a project and save its Project ID. Next,
create a service account as described
`here <https://cloud.google.com/bigquery/docs/reference/libraries#setting-up-authentitication>`__.
Download JSON file with your private key.

After that, you're good to go! ``get_comments`` function in RedditScore library
will help you to easily collect some Reddit comments. Let's say you want to collect
all comments posted in subreddits /r/BeardAdvice and /r/beards posted from
Jan 2016 to May 2016. This is how you do it:

>>> from redditscore import get_reddit_data as grd
>>> project_id = "my_first_project" # insert your Project ID here
>>> private_key = 'my_key.json' # insert path to your key file here
>>> subreddits = ['BeardAdvice', 'beards']
>>> df = grd.get_comments(subreddits, ('2016_01', '2016_05'),
project_id, private_key, verbose=True)

``df`` is a pandas DataFrame with collected comments. ``get_comments`` function
has a few additional options.

Saving results by month into CSV files instead of returning a DataFrame:

>>> grd.get_comments(subreddits, ('2016_01', '2016_05'),
project_id, private_key, verbose=True, csv_directory='reddit_data')

Getting a random sample of comments from each subreddit per month:

>>> df = grd.get_comments(subreddits, ('2016_01', '2016_05'),
project_id, private_key, verbose=True, comments_per_month=1000)

Getting top-scoring comments from each subreddit per month:

>>> df = grd.get_comments(subreddits, ('2016_01', '2016_05'),
project_id, private_key, verbose=True, comments_per_month=1000, top_scores=True)

Filtering by minimal score:

>>> df = grd.get_comments(subreddits, ('2016_01', '2016_05'),
project_id, private_key, verbose=True, score_limit=3)
