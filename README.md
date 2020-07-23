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
  * [Keras Models](#Keras-Models)
* [Scope of Improvement](#Scope-of-Improvement)
  * [Improving Surprise Models](#Improving-Surprise-Models)
  * [Improving Keras Models](#Improving-Keras-Models)
* [References](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Pipeline%20%26%20References.md)


## Overview

The following repository describes and evaluates various collaborative filtering models applied to users and video games on the Playstation Store. It also presents, and in some cases, explores avenues to improve the performance of said models.

## Packages
* [Python 3.7.4](https://docs.python.org/3.7/)
* [Beautful Soup 4.8.0](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#)
* [Keras 2.4.3](https://keras.io)
* [Surprise 1.1.0](https://surprise.readthedocs.io/en/stable/getting_started.html)
* [Hyperas 0.4.1](https://github.com/maxpumperla/hyperas)


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

</ol>

In both the above cases, besides tuning the hyperparameters, the train-test split method also played a major role in making more accurate predictions by eradicating the cold-start probelm. Another avenue I discovered to improve the results was to reduce the sparsity of the dataset itself. This is portrayed below.

![RMSE vs. Train size](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Img/trainsize.png?raw=true)

You can check out the Surprise models, predictions, top-n recommendations, and the reasoning behind the above hypotheses in the [MatrixFactorization_CollabFilter_PSN.ipynb](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/MatrixFactorization_CollabFilter_PSN.ipynb) notebook.


### Keras Models

Right off the bat, the Keras models provide much greater flexibility over the Surprise models in terms of architecture and the hyperparameters that can be tuned. The package allows you to build your own ANN architecture and fiddle with every aspect of it depending on the task at hand. Having a good understanding of the model and its hyperparameters can really help take advantage of this added flexibility. Some of the models I have built using this package are introduced below. 

<ol>
 <li><b>Multi-Layer Perceptron</b></li>
 
 <p>A multi-layer perceptron is essentially an ANN architecture that contains at least one hidden layer. In this particular case, users and items are assigned with a given number of embeddings that are treated as a set of features for that user or item. A generalized outline of such a model is shown below.</p>
 
 ![MLP Model](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Img/mlpmodel.png?raw=true)
 
 <p>This mode returned an MSE of 0.0762. One way to improve this model is to combine it with a matrix factorization model. This is exactly what the second keras Model defined in this project deals with.<p/>
 
 
 <li><b>Neural Matrix Factorization</b></li>
 
 <p>Simply put, this model combines the MLP model and the the matrix factorization model. It attempts to combine the MLP and MF frameworks so that they can mutually reinforce each other and learn more complex patterns in the data. A basic outline of this model is shown below.</p>
 
 ![NeuMF Model Architecure](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Img/neumfmodel.png?raw=true)
 
 <p>This model returned an MSE of 0.0751.</p>
 
 </ol>
 
<br>

The above images show highly simplified outlines of the model architecture. It is already clear from it that there are several aspects of it that can be modified, such as
 
 <ul>
  <li>Number of embeddings</li>
  <li>Number of hidden layers</li>
  <li>Number of neurons</li>
  <li>Layer activations</li>
  <li>The optimizer</li>
  <li>And so much more...</li>
 </ul>
 
In practice however, there are a lot more layers that fall in between what you see in the image above, and this introduces a large number of hyperparameters. You could imagine that optimizing such a large number of hyperparameters can get really time consuming. For this reason, I have used Hyperas, an optimization library, to save some time during this process and automate the tuning process to a certain extent.
 
The MLP model architure used in this project and its optimization can be found in the <a href = "https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Keras_Models/Hyperparameter_Optim/MLP_hyperasTune.py">MLP_hyperasTune.py</a> file under Keras_Models.

The NeuMF model architecture used in this project and its optimization can be found in the <a href = "https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Keras_Models/Hyperparameter_Optim/NeuMF_hyperasTune.py">NeuMF_hyperasTune.py</a> file under Keras_Models.

You can also check out how you can view the learning curves collected during the optimization processes for each model by visiting the <a href = "https://github.com/aneezJaheez/PSN-CF-Recommender-System/tree/master/Keras_Models/Optimization_logs">Optimization_logs</a> folder.</p>

The keras models built in the files linked above are described and studied in detail in the [Keras_CollabFilter_Models.ipynb](https://github.com/aneezJaheez/PSN-CF-Recommender-System/blob/master/Keras_CollabFilter_Models.ipynb) notebook. The notebook also describes the process of making predictions using the models and finding the top-n recommendations for each user in the system.


## Scope of Improvement

### Improving Surprise Models

As I have explained earlier, the Surprise package offers limited flexibility with the model architecture and optimization. Due to this trait, there are not many ways in which better results can be obtained using this package. 

The only logical conclusion I could arrive at in order to improve the results of this model was to obtain more data or decrease the sparsity of the dataset. Collaborative filtering is after all a model that depends on active user participation. Hence, having more ratings made per user and a larger number of ratings per game would be the best approach to improve recommendations made by this model.

### Improving Keras Models

The flexibility provided by keras in terms of model architecture open up a lot of avenues to improve the performance of the model. In addition to more active user participation in making video game ratings, here are some of the ways the models can be improved,

* Obtaining the timestamps for each individual user rating. This opens up a sequential aspect to the model which enables the use of Recurrent Neural Nets (RNN's) in making user rating predictions.
* Obtaining the review text from each user review.
