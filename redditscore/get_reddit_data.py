import calendar
import datetime
import os

import pandas as pd
from pandas.io import gbq


def diff_month(d1, d2):
    # Get difference between dates in months
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def add_months(sourcedate, months):
    # Add months to the date
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def construct_query(subreddits, month, score_limit):
    # Construct a query string
    subreddits = '", "'.join(subreddits)
    subreddits = '"' + subreddits + '"'
    if score_limit:
        score = " AND score >= {}".format(score_limit)
    else:
        score = ""
    query = """
    SELECT
        id,
        body,
        subreddit,
        author,
        created_utc,
        link_id,
        parent_id,
        score
    FROM [fh-bigquery:reddit_comments.""" + month + """]
    WHERE
        subreddit in (""" + subreddits + """)
        AND body != '[deleted]'
        AND body != '[removed]'
        AND body NOT LIKE '%has been removed%'
        AND body NOT LIKE '%has been overwritten%'
        AND body NOT LIKE '%performed automatically%'
        AND body NOT LIKE '%bot action performed%'
        AND body NOT LIKE '%autowikibot%'
        AND LENGTH(body) > 0""" + score
    return query


def construct_sample_query(subreddits, month, sample_size, score_limit):
    # Constuct a query string with random sampling
    subreddits = '", "'.join(subreddits)
    subreddits = '"' + subreddits + '"'
    if score_limit:
        score = " AND score >= {}".format(score_limit)
    else:
        score = ""
    query = """
    SELECT
        id,
        body,
        subreddit,
        author,
        created_utc,
        link_id,
        parent_id,
        score
    FROM (
        SELECT
            id,
            body,
            subreddit,
            author,
            created_utc,
            link_id,
            parent_id,
            score,
            RAND() as rnd,
            ROW_NUMBER() OVER(PARTITION BY subreddit ORDER BY rnd) as pos
        FROM [fh-bigquery:reddit_comments.""" + month + """]
        WHERE
            subreddit in (""" + subreddits + """)
            AND body != '[deleted]'
            AND body != '[removed]'
            AND body NOT LIKE '%has been removed%'
            AND body NOT LIKE '%has been overwritten%'
            AND body NOT LIKE '%performed automatically%'
            AND body NOT LIKE '%bot action performed%'
            AND body NOT LIKE '%autowikibot%'
            AND LENGTH(body) > 0""" + score + """
    )
    WHERE pos <= """ + str(sample_size)
    return query


def get_comments(subreddits, timerange, project_id, private_key, score_limit=0,
                 comments_per_month=None, csv_directory=None, verbose=False):
    """
    Obtain Reddit comments using Google BigQuery

    Parameters
    ----------
    subreddits: list
        List of subreddit names

    timerange: iterable, shape (2,)
        Start and end dates in the '%Y_%m' format.
        Example: ('2016_08', '2017_02')

    project_id: str
        Google BigQuery Account project ID

    private_key: str
        File path to JSON file with service account private key
        https://cloud.google.com/bigquery/docs/reference/libraries

    score_limit: int, optional
        Score limit for comment retrieving. If None, retrieve all comments.

    comments_per_month: int, optional
        Number of comments to sample from each subbredit per month. If None,
        retrieve all comments.

    csv_directory: str, optional
        CSV directory to save retrieved data. If None, return a DataFrame with
        all comments.

    verobse: bool, optional
        If True, print the name of the table, which is being queried.

    Returns
    ----------
    df: pandas DataFrame
        DataFrame with comments.
    """
    if not isinstance(subreddits, list):
        raise ValueError(
            'subreddits argument must be a list, not {}'.format(type(subreddits)))
    if (comments_per_month is not None) and \
            not isinstance(comments_per_month, int):
        raise ValueError('comments_per_month must be an integer, not {}'.format(
            type(comments_per_month)))
    if (csv_directory is not None) and \
            not os.path.isdir(csv_directory):
        raise OSError('{} does not exist'.format(csv_directory))
    try:
        iter(timerange)
    except TypeError as e:
        raise TypeError('timerange argument must be an iterable') from e
    try:
        assert len(timerange) == 2
    except AssertionError as e:
        raise ValueError(
            'timerange argument has to contain only two elements') from e

    start = datetime.datetime.strptime(timerange[0], '%Y_%m')
    end = datetime.datetime.strptime(timerange[1], '%Y_%m')
    delta = diff_month(end, start)
    if csv_directory is None:
        dfs = []
    else:
        dfs = None

    for i in range(delta + 1):
        date = add_months(start, i)
        table_name = date.strftime('%Y_%m')
        if verbose:
            print(
                'Querying from [fh-bigquery:reddit_comments.{}]'.format(table_name))
        if comments_per_month is None:
            query = construct_query(subreddits, table_name, score_limit)
        else:
            query = construct_sample_query(
                subreddits, table_name, comments_per_month, score_limit)
        df = gbq.read_gbq(query, project_id=project_id,
                          private_key=private_key)
        if csv_directory is None:
            dfs.append(df)
        else:
            df.to_csv(os.path.join(csv_directory,
                                   table_name + '.csv'), index=False)
    return dfs
