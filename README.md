# PS Store Recommender System

A collaborative filtering based video game recommender system for users of the Playstation 4.


## Index
* [Overview](#Overview)
* [Packages](#Packages)
* [The Dataset](#The-Dataset)
  * [Source](#Source)
  * [Key Features](#Key-Features)
* [The Models]


## Overview

The following repository describes and evaluates various collaborative filtering models applied to users and video games on the Playstation Store. It also presents, and in some cases, explores avenues to improve the performance of said models.

## Packages
* [Python 3.7.4](https://docs.python.org/3.7/)
* [Beautful Soup 4.8.0](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#)
* [Keras 2.4.3](https://keras.io)
* [Surprise 1.1.0](https://surprise.readthedocs.io/en/stable/getting_started.html)


## The Dataset

### Source

The data used in this model has been scraped from a popular game review site called [Metacritic](https://www.metacritic.com). The scraping algorithm has been built using the Python Beautiful Soup library.

The data obtained from the website includes the users and their respective reviews for all the video games on the Playstation 4.


### Key Features

The dataset contains users, games, the ratings as 3 respective columns. Some of the key features of the dataset are highlighted below.
1. Number of users : 113,339
2. Number of video games : 1,724
3. Sparisity : 0.089%
4. Shape : (171858, 3)
5. Number of Users with over 20 ratings :  325
6. Number of Games with over 20 ratings :  595
7. The Average rating of games do not drastically increase with the increase in popularity.

![Average Rating vs. No. of Ratings](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Img/avgRatingVnum.png?raw=true)

Further exploration and analysis has been carried out in the "ExploratoryAnalysis.ipynb" notebook.

