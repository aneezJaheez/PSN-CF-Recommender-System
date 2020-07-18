#Python script to build a deep learning keras model for collaborative filtering and tune the hyperparameters for best results using hyperas

#Keras Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks

#Hyperas Imports
from hyperas.distributions import uniform
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim

#Other essential libraries
import pandas as pd
import numpy as np

#Model function definition for hyperas parameter tuning

def model(X_train, y_train, X_valid, y_valid):
    
    #Choosing the number of embeddings for the users and items. 
    num_embed_user = {{choice([30, 40, 50, 60])}}
    num_embed_game = {{choice([20, 30, 40, 50])}}
    
    #Choosing optimal values for ridge regularization in the embeddings layer
    l2_param = {{uniform(0, 1e-3)}}
    
    
    #Selecting the optimization to be tested
    optval = {{choice(["sgd", "adam"])}}
    
    #Choosing the learning rates and deining the optimization function
    if(optval == "sgd"):
        sgd_lr_param = {{uniform(0, 0.1)}}
        optim = keras.optimizers.SGD(lr = sgd_lr_param, momentum = {{uniform(0, 1)}})
    else:
        adam_lr_param = {{uniform(1e-7, 1e-3)}}
        optim = keras.optimizers.Adam(lr = adam_lr_param)
    
    #Number of neurons per hidden layer. I will be using the same number of neurons in every hidden layer
    num_neurons = {{choice([4, 8, 16, 32, 64, 128])}}
    
    #Number of unique users and unique games to for embedding layer specs
    num_users = len(X_train["User_Enc"].unique())
    num_games = len(X_train["Game_Enc"].unique())
    
    #DL ARCHITECUTURE DEFINITION
    #===========================================================================================================
    
    #1st Input for the network which takes in the unique integer encodings given to the user
    user_input = layers.Input(shape = (1, ), name = "User_Name_Input")
    #Embedding layer to create embedding vectors for each user with the chosen length
    user_embed = layers.Embedding(
        num_users, 
        num_embed_user, 
        embeddings_initializer='he_normal', 
        embeddings_regularizer=keras.regularizers.l2(l2_param), 
        name = "User_Embeddings"
    )(user_input)
    #Flattening the embedding layer so that it can be fed into a dense layer.
    #Reshaping or flattening the embedding layer is crucial before feeding it into a dense layer.
    user_embed_flat = layers.Flatten(name = "Flat_User_Embeddings")(user_embed)

    #2nd Input for the network which takes in the unique integer encodings given to each game 
    game_input = layers.Input(shape = (1, ), name = "Game_Name_Input")
    #Embedding layer to create embeddings for each game with the specified length
    game_embed = layers.Embedding(
        num_games, 
        num_embed_game, 
        embeddings_initializer='he_normal', 
        embeddings_regularizer=keras.regularizers.l2(l2_param),
        name = "Game_Embeddings"
    )(game_input)
    #Flattening the output from the embeddings layer before feeding it into the dense layer
    game_embed_flat = layers.Flatten(name = "Flat_Game_Embeddings")(game_embed)
    
    #Now the embeddings mimic a set of features that can be used as inputs into the DL model
    
    #Concatenating the user and game embeddings. The output from this layer will be the input into the hidden layers
    concat = layers.Concatenate(name = "User_Game_Embeddings")([game_embed_flat, user_embed_flat])
    
    
    #Hidden layer definitions
    
    #All hidden layers will be using the same activation function that is recommended for regression-
    #since collaborative filtering is essentially a multiple regression problem over multiple users and games
    
    #Hidden layer 1
    dense1 = layers.Dense(num_neurons, activation = "relu", name = "Hidden_Layer_1")(concat)
    #Batch normalization for hidden layer 1 to help the model converge faster
    dense1 = layers.BatchNormalization(name='batch_norm1')(dense1)
    
    #Dropout layer to prevent overfitting. The dropout layer sets random paramters to 0 at the specified frequency
    dense1 = (layers.Dropout({{uniform(0, 1)}}, name = "Dropout_Layer_1")(dense1))
    
    num_extra_layers = {{choice([0, 1])}}
    
    if(num_extra_layers == 1):
        dense2 = layers.Dense(num_neurons, activation = "relu", name = "Hidden_Layer_2")(dense1)
        dense2 = layers.BatchNormalization(name = "batch_norm2")(dense2)
        dense2 = (layers.Dropout({{uniform(0, 1)}}, name = "Dropout_Layer_1")(dense2))
        
        #Output layer
        output = layers.Dense(1, activation = "relu", name = "Output_Layer")(dense2)
    else:
        #Output layer
        output = layers.Dense(1, activation = "relu", name = "Output_Layer")(dense1)
    
    #Constraining the outputs to the range [0, 1] since we have normalized our outputs to this range
    result = keras.activations.relu(output, max_value = 1.0, threshold = 0.0)
    
    
    #===============================================================================================================
    #END OF MODEL ARCHITECTURE
    
    #Initializing the model using the above architecture
    model = keras.Model(inputs=[game_input, user_input], outputs=result)

    #Early stopping callback which will stop the learning process when there is insignificant improvement between each iteration.
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor = "val_loss", #Monitors the validation set error
        min_delta = 0.002, #Minimum value to be considered as a significant improvement
        patience = 1, #Number of iterations to continue running with insignificant improvement
        restore_best_weights = True, #Use parameters from the best iteration in the model 
    )
    
    #Model compilation
    model.compile(loss='mse', optimizer=optim, metrics = ['accuracy'])
    
    #Training and validating the model
    model.fit(
        x = [X_train.Game_Enc, X_train.User_Enc], 
        y=y_train.Rating, 
        batch_size = 128, 
        epochs=100, #A high value can be used sicne early stopping is enabled
        verbose=1, 
        validation_data=([X_valid.Game_Enc, X_valid.User_Enc], y_valid.Rating), 
        callbacks = early_stopping_cb
    )
    
    #Evaluating the model on the validation set using the optimal hyperparameters and saving the mse(loss)
    loss, acc = model.evaluate([X_valid.Game_Enc, X_valid.User_Enc], y_valid.Rating)
    
    return {'loss': loss, 'status': STATUS_OK, 'model': model}



#data retrieval and processing function for hyperas parameter tuning
def data():
    """Note that the data has to be retrieved within this function even if it has retrieved 
       already at some other point in the script."""
    
    #Importing the train and validation sets
    trainset = pd.read_csv("trainset.csv")
    validset = pd.read_csv("validset.csv")   

    #Preparing the train and validation sets
    X_train = trainset[["User_Enc", "Game_Enc"]]
    y_train = trainset[["Rating"]]
    X_valid = validset[["User_Enc", "Game_Enc"]]
    y_valid = validset[["Rating"]]
    
    return X_train, y_train, X_valid, y_valid  


#Execution starts here
if __name__ == '__main__':

	#Model optimization using the above defined functions
    best_run, best_model = optim.minimize(
    	model=model,
        data=data,
        algo=tpe.suggest,
        max_evals=20,
        trials=Trials(),
        eval_space = True
    )

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    #Saving the best model so it can be imported in the notebook
    best_model.save("Keras_Collab_DL_Model.h5")


#End of hyperparameter tuning process
