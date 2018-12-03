# -*- coding: utf-8 -*-
"""
Tools to collect Twitter data from specific accounts.
Part of the module is based on Twitter Scraper library:
https://github.com/bpb27/twitter_scraping

Author: Evgenii Nikitin <e.nikitin@nyu.edu>

Part of https://github.com/crazyfrogspb/RedditScore project

Copyright (c) 2018 Evgenii Nikitin. All rights reserved.
This work is licensed under the terms of the MIT license.
"""

import datetime
import math
import os.path as osp
import warnings
from time import sleep

import pandas as pd
from dateutil import parser

import tweepy
from congress import Congress

try:
    from selenium import webdriver
    from selenium.common.exceptions import (NoSuchElementException,
                                            StaleElementReferenceException,
                                            WebDriverException)
except ImportError:
    warnings.warn(('selenium library is not found, pulling tweets beyond'
                   ' 3200 limit will be unavailable'))


def _format_day(date):
    # convert date to required format
    day = '0' + str(date.day) if len(str(date.day)) == 1 else str(date.day)
    month = '0' + str(date.month) if len(str(date.month)
                                         ) == 1 else str(date.month)
    year = str(date.year)
    return '-'.join([year, month, day])


def _form_url(since, until, user):
    # create url request
    p1 = 'https://twitter.com/search?f=tweets&vertical=default&q=from%3A'
    p2 = user + '%20since%3A' + since + '%20until%3A' + \
        until + 'include%3Aretweets&src=typd'
    return p1 + p2


def _increment_day(date, i):
    # increment date by i days
    return date + datetime.timedelta(days=i)


def _grab_tweet_by_ids(ids, api, delay=6.0):
    # grab tweets by ids
    full_tweets = []
    start = 0
    end = 100
    limit = len(ids)
    i = math.ceil(limit / 100)

    for go in range(i):
        sleep(delay)
        id_batch = ids[start:end]
        start += 100
        end += 100
        tweets = api.statuses_lookup(id_batch, tweet_mode='extended')
        full_tweets.extend(tweets)

    return full_tweets


def _grab_even_more_tweets(screen_name, dates, browser, delay=1.0):
    # grab tweets beyond 3200 limit
    startdate, enddate = dates

    try:
        if browser == 'Safari':
            driver = webdriver.Safari()
        elif browser == 'Firefox':
            driver = webdriver.Firefox()
        elif browser == 'Chrome':
            driver = webdriver.Chrome()
        else:
            raise ValueError('{} browser is not supported')
    except WebDriverException as e:
        raise WebDriverException(('You need to download required driver'
                                  ' and add it to PATH')) from e
    except AttributeError as e:
        raise Exception('Check if the browser is installed') from e
    except ValueError as e:
        raise ValueError('{} browser is not supported') from e

    days = (enddate - startdate).days + 1
    id_selector = '.time a.tweet-timestamp'
    tweet_selector = 'li.js-stream-item'
    screen_name = screen_name.lower()
    ids = []

    for day in range(days):
        d1 = _format_day(_increment_day(startdate, 0))
        d2 = _format_day(_increment_day(startdate, 1))
        url = _form_url(d1, d2, screen_name)
        driver.get(url)
        sleep(delay)

        try:
            found_tweets = driver.find_elements_by_css_selector(tweet_selector)
            increment = 10

            while len(found_tweets) >= increment:
                driver.execute_script(
                    'window.scrollTo(0, document.body.scrollHeight);')
                sleep(delay)
                found_tweets = driver.find_elements_by_css_selector(
                    tweet_selector)
                increment += 10

            for tweet in found_tweets:
                try:
                    id = tweet.find_element_by_css_selector(
                        id_selector).get_attribute('href').split('/')[-1]
                    ids.append(id)
                except StaleElementReferenceException as e:
                    pass
        except NoSuchElementException:
            pass

        startdate = _increment_day(startdate, 1)

    return ids


def _handle_tweepy_error(e, user):
    if e.api_code == 34:
        warnings.warn("{} doesn't exist".format(user))
    else:
        warnings.warn('Error encountered for user {}: '.format(
            user) + str(e))
    return pd.DataFrame()


def generate_api(twitter_creds_list):
    auths = []
    for creds in twitter_creds_list:
        try:
            auth = tweepy.OAuthHandler(
                creds['consumer_key'], creds['consumer_secret'])
            auth.set_access_token(creds['access_key'], creds['access_secret'])
        except KeyError as e:
            raise Exception(("twitter_creds must contain cosnumer_key,"
                             " consumer_secret, access_key, and access_secret keys"))
        auths.append(auth)
    api = tweepy.API(
        auths,
        retry_count=3,
        retry_delay=5,
        retry_errors=set([401, 404, 500, 503]),
        monitor_rate_limit=True,
        wait_on_rate_limit=True,
        wait_on_rate_limit_notify=True)
    return api


def grab_tweets(twitter_creds=None, api=None, screen_name=None, user_id=None, timeout=0.1,
                fields=None, get_more=False, browser='Firefox',
                start_date=None):
    """
    Get all tweets from the account

    Parameters
    ----------
    twitter_creds: dict
        Dictionary or list with Twitter authentication credentials.
        Has to contain consumer_key, consumer_secret, access_key, access_secret

    screen_name : str, optional
        Twitter handle to grab tweets for

    user_id: int, optional
        Twitter user_id to grab tweets for

    timeout: float, optional
        Sleeping time between requests

    fields: iter, optional
        Extra fields to pull from the tweets

    get_more: bool, optional
        If True, attempt to use Selenium to get more tweets after reaching
        3200 tweets limit

    browser: {'Firefox', 'Chrome', 'Safari'}, optional
        Browser for Selenium to use. Corresponding browser and its webdriver
        have to be installed

    start_date: datetime.date, optional
        The first date to start pulling extra tweets. If None, use 2016/01/01

    Returns
    ----------
    alltweets: pandas DataFrame
        Pandas Dataframe with all collected tweets
    """
    if not (bool(screen_name) != bool(user_id)):
        raise ValueError('You have to provide either screen_name or user_id')

    api = generate_api(list(twitter_creds))

    if user_id:
        try:
            u = api.get_user(int(user_id))
            screen_name = u.screen_name
            reg_date = u.created_at.date()
            sleep(timeout)
        except tweepy.TweepError as e:
            return _handle_tweepy_error(e, user_id)
        except ValueError as e:
            raise ValueError('{} is not a valid user_id'.format(user_id)) from e
    else:
        u = api.get_user(screen_name)
        reg_date = u.created_at.date()
        sleep(timeout)

    if fields is None:
        fields = []

    if start_date is None or start_date < reg_date:
        start_date = reg_date

    alltweets = []

    print("Now grabbing tweets for {}".format(screen_name))
    try:
        new_tweets = api.user_timeline(screen_name=screen_name,
                                       user_id=user_id, count=200,
                                       tweet_mode='extended')
    except tweepy.TweepError as e:
        return _handle_tweepy_error(e, screen_name)

    alltweets.extend(new_tweets)
    if not alltweets:
        return pd.DataFrame()
    oldest = alltweets[-1].id - 1

    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name=screen_name, count=200,
                                       max_id=oldest, tweet_mode='extended')
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        if new_tweets:
            print('{} tweets downloaded'.format(len(alltweets)))
        sleep(timeout)

    if get_more and len(new_tweets) == 0 and len(alltweets) > 3200:
        end_date = alltweets[-1].created_at.date()
        print('Date of the last collected tweet: {}'.format(end_date))
        if end_date > start_date:
            print('Now grabbing tweets beyond 3200 limit')
            dates = (start_date, end_date)
            ids = _grab_even_more_tweets(screen_name, dates, browser)
            tweets = _grab_tweet_by_ids(ids, api)
            alltweets.extend(tweets)

    full_tweets = []
    for tweet in alltweets:
        if hasattr(tweet, 'retweeted_status'):
            text = tweet.retweeted_status.full_text
        else:
            text = tweet.full_text
        retweet = False
        if getattr(tweet, 'retweeted_status', None) is not None:
            retweet = True
        tweet_fields = [text, tweet.id, tweet.created_at, retweet]
        for field in fields:
            tweet_fields.append(getattr(tweet, field, None))
        full_tweets.append(tweet_fields)
    full_tweets = pd.DataFrame(
        full_tweets, columns=(['text', 'id', 'created_at', 'retweet'] +
                              fields))
    full_tweets['screen_name'] = screen_name
    if user_id:
        full_tweets['user_id'] = user_id
    full_tweets.drop_duplicates('id', inplace=True)
    return full_tweets


def collect_congress_tweets(congress_list, congress_tweets_file,
                            meta_info_file, start_date, twitter_creds,
                            chambers=None, propublica_api_key=None,
                            append_frequency=10, browser='Chrome',
                            fields=None):
    """Collect tweets from American Congressmen.

    Parameters
    ----------
    congress_list : iterable
        List with Congress numbers to collect data for.
    congress_tweets_file : str
        Path to the output file with tweets.
    meta_info_file : str
        Path to the output file with meta information about the Congress.
    start_date : str
        The first date to start pulling extra tweets.
    twitter_creds : type
        Dictionary or list with Twitter authentication credentials.
        Has to contain consumer_key, consumer_secret, access_key, access_secret
    chambers : iterable, optional
        List of Chambers to collect tweets for (the default is Senate and House).
    propublica_api_key : str, optional
        API key for free Propublica Congress API (the default is None).
        https://www.propublica.org/datastore/api/propublica-congress-api
    append_frequency : int, optional
        Frequency of dumping new tweets to CSV (the default is 10).
    browser : type
        Browser for Selenium to use. Corresponding browser and its webdriver
        have to be installed (the default is 'Chrome').
    fields : type
        Extra fields to pull from the tweets (the default is retweet_count and favorite_count).
    """
    if chambers is None:
        chambers = ['House', 'Senate']
    if fields is None:
        fields = ['retweet_count', 'favorite_count']

    if osp.isfile(meta_info_file):
        members = pd.read_csv(meta_info_file)
    else:
        congress = Congress(propublica_api_key)
        all_members = []
        for congress_num in congress_list:
            for chamber in chambers:
                members = pd.DataFrame(congress.members.filter(
                    chamber, congress=congress_num)[0]['members'])
                members['chamber'] = chamber
                members['congress_num'] = congress_num
                all_members.append(members)
        members = pd.concat(all_members)
        members.to_csv(meta_info_file, index=False)

    twitter_handles = members.twitter_account.unique()
    start_date = parser.parse(start_date).date()
    if osp.isfile(congress_tweets_file):
        tweets = pd.read_csv(congress_tweets_file,
                             lineterminator='\n', usecols=['screen_name'])
        parsed_handles = list(tweets['screen_name'].unique())
        del tweets
    else:
        parsed_handles = []

    dfs = []
    for i, twitter_handle in enumerate(twitter_handles):
        if twitter_handle in parsed_handles or pd.isnull(twitter_handle):
            continue

        try:
            df = grab_tweets(twitter_creds, screen_name=twitter_handle, timeout=1.0,
                             get_more=True, start_date=start_date, browser=browser, fields=fields)
        except Exception as e:
            warnings.warn(f'Exception occured for {twitter_handle}: {e}')
            continue

        parsed_handles.append(twitter_handle)
        if df.empty:
            continue
        df = df.loc[df.created_at >= pd.Timestamp(start_date)]
        dfs.append(df)
        if len(dfs) == append_frequency or i == (len(twitter_handles) - 1):
            df = pd.concat(dfs)
            if osp.isfile(congress_tweets_file):
                df.to_csv(congress_tweets_file, mode='a',
                          header=False, index=False)
            else:
                df.to_csv(congress_tweets_file, index=False)
            dfs = []
