# PS Store Recommender System

A collaborative filtering based video game recommender system for users of the Playstation 4.


## Index
* [Overview](#Overview)
* [Packages](#Packages)
* [The Dataset](#The-Dataset)
  * [Source](#Source)
  * [Key Features](#Key-Features)
* [The Models](#The-Models)
 * [Surprise Models](#Surprise-Models)


## Overview

The following repository describes and evaluates various collaborative filtering models applied to users and video games on the Playstation Store. It also presents, and in some cases, explores avenues to improve the performance of said models.

## Packages
* [Python 3.7.4](https://docs.python.org/3.7/)
* [Beautful Soup 4.8.0](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#)
* [Keras 2.4.3](https://keras.io)
* [Surprise 1.1.0](https://surprise.readthedocs.io/en/stable/getting_started.html)


## The Dataset

### Source

The data used in this model has been scraped from a popular game review site called [Metacritic](https://www.metacritic.com). The scraping algorithm has been built using the Python Beautiful Soup library. Click [here](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/metacriticScraper.py) to check out the code.

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

Further exploration and analysis has been carried out in the [ExploratoryAnalysis.ipynb](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/ExploratoryAnalysis.ipynb) notebook.

## The Models

The models used in this project have been derived from two packages; Python Surprise and Keras.

### Surprise Models

<ol>
 <a href = "https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD"><li>Singular Value Decomposition (SVD)</li></a>
 
 <p>A matrx factorization based model for collaborative filtering that makes recommendation based on a set of hidden features found for each user and item. The model provides limited flexibility in terms of tuning hyperparameters by enabling modifications to the Learning rate, Regularization, and Epochs. All of these hyperparameters have been tuned manually with the help of learing curves.</p>
 
 ![Learning Rate vs. RMSE](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Img/Alpha.png?raw=true)
 ![Regularization vs. RMSE](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Img/Lambda.png?raw=true)
 ![Epochs vs. RMSE](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Img/epochs.png?raw=true)
 
 <p>This model returned an MSE of 0.070.</p>

 
 <a href = "https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly"><li>Baseline Alternating Least Squares (ALS)</li></a>
 
 <p>While SVD uses Stochastic Gradient Descent(SGD) to converge to optimal results, the baseline algorithm uses Alternating Least Squares. ALS usually converges to better results than SGD when dealing with sparse datasets. However, the improvement in this case was not significant. It is less flexible to tuning hyperparameters than SVD as it only allows modifications to the regularization and epochs. The best combination of hyperparameters in this case has been found using the GridSearchCV function in the surprise package.</p>
 
 ```python
from surprise.model_selection import GridSearchCV
#Setting up the range of hyperparameters
param_grid = {
    'bsl_options' : {
        'method' : ['als'],
        'reg_u' : [0.03, 0.09, 0.1, 0.3, 0.9, 1, 3, 9],
        'reg_i' : [0.03, 0.09, 0.1, 0.3, 0.9, 1, 3, 9],
        'n_epochs' : [15, 20, 25]
    }
}

#Finding the optimal combination of parameters
gs_als = GridSearchCV(BaselineOnly, param_grid, measures=['rmse'], cv=3)
```

<p>Using the ALS model, I was able to achieve a slightly better MSE of 0.0676</p>

<p>Besides tuning the hyperparameters, the train-test split method also played a major role in making more accurate predictions by eradicating the cold-start probelm. Another avenue I discovered to improve the results was to reduce the sparsity of the dataset itself. This is portrayed below.</p>

![RMSE vs. Train size](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Img/trainsize.png?raw=true)

You can check out the model, predictions, top-n recommendations, and the reasoning behind the above hypotheses in the [MatrixFactorization_CollabFilter_PSN.ipynb](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/MatrixFactorization_CollabFilter_PSN.ipynb) notebook.

</ol>
