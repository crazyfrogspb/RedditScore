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
from time import sleep

import pandas as pd
import tweepy

try:
    from selenium import webdriver
    from selenium.common.exceptions import (NoSuchElementException,
                                            StaleElementReferenceException,
                                            WebDriverException)
except ImportError:
    warnings.warn(('selenium library is not found, pulling tweets beyond'
                   ' 3200 limit will be unavailable'))


def format_day(date):
    # convert date to required format
    day = '0' + str(date.day) if len(str(date.day)) == 1 else str(date.day)
    month = '0' + str(date.month) if len(str(date.month)
                                         ) == 1 else str(date.month)
    year = str(date.year)
    return '-'.join([year, month, day])


def form_url(since, until, user):
    # create url request
    p1 = 'https://twitter.com/search?f=tweets&vertical=default&q=from%3A'
    p2 = user + '%20since%3A' + since + '%20until%3A' + \
        until + 'include%3Aretweets&src=typd'
    return p1 + p2


def increment_day(date, i):
    # increment date by i days
    return date + datetime.timedelta(days=i)


def grab_tweet_by_ids(ids, api, delay=6.0):
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


def grab_even_more_tweets(screen_name, dates, browser, delay=1.0):
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
                                  ' and add it to path')) from e
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
        d1 = format_day(increment_day(startdate, 0))
        d2 = format_day(increment_day(startdate, 1))
        url = form_url(d1, d2, screen_name)
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

        startdate = increment_day(startdate, 1)

    return ids


def grab_tweets(screen_name, twitter_creds, timeout=0.1, fields=None,
                get_more=False, browser='Firefox', start_date=None):
    """
    Get all tweets from the account

    Parameters
    ----------
    screen_name : str
        Twitter handle to grab tweets for

    twitter_creds: dict
        Dictionary with Twitter authentication credentials.
        Has to contain consumer_key, consumer_secret, access_key, access_secret

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
    try:
        auth = tweepy.OAuthHandler(
            twitter_creds['consumer_key'], twitter_creds['consumer_secret'])
        auth.set_access_token(
            twitter_creds['access_key'], twitter_creds['access_secret'])
        api = tweepy.API(auth)
    except KeyError as e:
        raise Exception(("twitter_creds must contain cosnumer_key,"
                         " consumer_secret, access_key, and access_secret keys"))

    if fields is None:
        fields = []
    if start_date is None:
        start_date = datetime.date(year=2016, month=1, day=1)

    alltweets = []

    print("Now grabbing tweets for {}".format(screen_name))
    new_tweets = api.user_timeline(
        screen_name=screen_name, count=200, tweet_mode='extended')
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1

    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name=screen_name, count=200,
                                       max_id=oldest, tweet_mode='extended')
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        print('{} tweets downloaded'.format(len(alltweets)))
        sleep(timeout)

    if get_more and len(new_tweets) == 0 and len(alltweets) > 3200:
        print('Now grabbing tweets beyond 3200 limit')
        end_date = alltweets[-1].created_at.date()
        if end_date > start_date:
            dates = (start_date, end_date)
            ids = grab_even_more_tweets(screen_name, dates, browser)
            tweets = grab_tweet_by_ids(ids, api)
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
        tweet_fields = [text, tweet.id_str, tweet.created_at, retweet]
        for field in fields:
            tweet_fields.append(getattr(tweet, field, None))
        full_tweets.append(tweet_fields)
    full_tweets = pd.DataFrame(
        full_tweets, columns=(['text', 'id_str', 'created_at', 'retweet'] +
                              fields))
    full_tweets['screen_name'] = screen_name
    full_tweets.drop_duplicates('id_str', inplace=True)
    return full_tweets
