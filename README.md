# PS Store Recommender System

A collaborative filtering based video game recommender system for users of the Playstation 4.


## Index
* [Overview](#Overview)
* [Packages](#Packages)
* [The Dataset](#The-Dataset)


## Overview

The following repository describes and evaluates various collaborative filtering models applied to users and video games on the Playstation Store. It also presents, and in some cases, explores avenues to improve the performance of said models.

## Packages
* [Python 3.7.4](https://docs.python.org/3.7/)
* [Beautful Soup 4.8.0](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#)
* [Keras 2.4.3](https://keras.io)
* [Surprise 1.1.0](https://surprise.readthedocs.io/en/stable/getting_started.html)


## The Dataset

The data used in this model has been scraped from a popular game review site called [Metacritic][https://www.metacritic.com]. The scraping algorithm has been built using the Python Beautiful Soup library.

The data obtained from the website includes the users and their respective reviews for all the video games on the Playstation 4.

