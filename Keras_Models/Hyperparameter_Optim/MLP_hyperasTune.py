#Python script to build a deep learning keras model for collaborative filtering and tune the hyperparameters for best results using hyperas

#Keras Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks

#Hyperas Imports
from hyperas.distributions import uniform
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim

#Other essential libraries
import pandas as pd
import numpy as np

#Imports for logging runs
import os

#Model function definition for hyperas parameter tuning

def model(X_train, y_train, X_valid, y_valid):
    """
    Model Function to define the network architecture and hyperparameters.
    This function will be passed to the hyperas optim function for hyperparameter optimization.

    Arguments : 
    -> Trainset inputs (X_train)
    -> Trainset Outputs (y_train)
    -> Validation Inputs (X_valid)
    -> Validation Outputs (y_valid)

    Returns:
    -> Error metric (Loss in this case)
    -> Status
    -> Keras Model

    """

    #Logging the runs during hyperparameter tuning

    #log directory setting
    root_logdir = os.path.join(os.curdir, "mlp_run_logs")
    
    #Logging function to name and return the folder to be used
    #The logs will be named with the convention "run_(num_user_embeddings)_(num_game_embeddings)_(num_neurons)_(num_hidden_layers)_(optimization_function)_(learning_rate)"
    def get_run_logdir(emb_u, emb_g, l2, neurons, layers, opt, lr): 
        import time
        run_id = "run_" + str(emb_u) + "_" + str(emb_g) + "_" + str(neurons) + "_" + str(layers) + "_" + opt + "_" + lr
        return os.path.join(root_logdir, run_id)


    #Choosing the number of embeddings for the users and items. 
    num_embed_user = {{choice([30, 40, 50, 60])}}
    num_embed_game = {{choice([20, 30, 40, 50])}}
    
    #Optimal value from range for ridge regularization in the embeddings layer
    l2_param = {{uniform(3e-8, 1e-6)}}
    
    #Selecting the optimization to be tested
    optval = {{choice(["sgd", "adam"])}}
    
    #Choosing the learning rates and deining the optimization function based on optval value
    if(optval == "sgd"):
        sgd_lr_param = {{uniform(0, 0.1)}}
        optim = keras.optimizers.SGD(lr = sgd_lr_param, momentum = {{uniform(0, 1)}})
    else:
        adam_lr_param = {{uniform(3e-6, 1e-4)}}
        optim = keras.optimizers.Adam(lr = adam_lr_param)
    
    #Number of neurons per hidden layer. I will be using the same number of neurons in every hidden layer
    num_neurons = {{choice([4, 8, 16, 32, 64, 128])}}
    
    #Number of unique users and unique games used for embedding layer specs
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
        embeddings_initializer = 'he_normal', 
        embeddings_regularizer = keras.regularizers.l2(l2_param), 
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
    
    #Now the embeddings mimic a set of features that can be used as inputs into the DL model after concatenation.
    concat = layers.Concatenate(name = "User_Game_Embeddings")([game_embed_flat, user_embed_flat])
    
    
    #Hidden layer definitions
    
    #All hidden layers will be using the same activation function(relu) that is recommended for regression-
    #since collaborative filtering is essentially a multiple regression problem over multiple users and games
    
    #Hidden layer 1
    dense1 = layers.Dense(num_neurons, activation = "relu", name = "Hidden_Layer_1")(concat)
    #Batch normalization for hidden layer 1 to help the model converge faster
    #I have not used normalization in this case because time is not a constraint during tuning
    # dense1 = layers.BatchNormalization(name='batch_norm1')(dense1)
    #Dropout layer to prevent overfitting. The dropout layer sets random paramters to 0 at the specified frequency
    dense1 = (layers.Dropout({{uniform(0, 0.4)}}, name = "Dropout_Layer_1")(dense1))
    
    #For most cases, 1 hidden layer with sufficient number of neurons should be able to fit the data well. 
    #Option to add extra hidden layers
    num_extra_layers = {{choice([0, 1])}}
    
    if(num_extra_layers == 1):
        #Defining second hidden layer with normalization and dropout
        dense2 = layers.Dense(num_neurons, activation = "relu", name = "Hidden_Layer_2")(dense1)
        # dense2 = layers.BatchNormalization(name = "batch_norm2")(dense2)
        dense2 = (layers.Dropout({{uniform(0, 0.4)}}, name = "Dropout_Layer_2")(dense2))
        
        #Output layer with a single neuron for 1 output per user-item pair
        output = layers.Dense(1, activation = "relu", name = "Output_Layer")(dense2)
    else:
        #Output layer
        output = layers.Dense(1, activation = "relu", name = "Output_Layer")(dense1)
    
    #Constraining the outputs to the range [0, 1] since we have normalized our outputs to this range
    result = keras.activations.relu(output, max_value = 1.0, threshold = 0.0)
    
    
    #===============================================================================================================
    #END OF MODEL ARCHITECTURE
    
    #Initializing the model using the above architecture
    model = keras.Model(inputs=[game_input, user_input], outputs=output)

    #Early stopping callback which will stop the learning process when there is insignificant improvement between each iteration.
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor = "val_loss", #Monitors the validation set error
        min_delta = 15e-4, #Minimum value to be considered as a significant improvement
        patience = 1, #Number of iterations to continue running with insignificant improvement
        restore_best_weights = True, #Use parameters from the best iteration in the model 
    )

    
    #Learning rate that will be passed to the log directory
    if(optval == "sgd"):
        lr = (str(sgd_lr_param))[:5]
    else:
        lr = (str(adam_lr_param))[:4] + (str(adam_lr_param))[-4:]

    #Defining callback to log optimization runs
    run_logdir = get_run_logdir(num_embed_user, num_embed_game, l2_param, num_neurons, num_extra_layers, optval, lr)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    #Model compilation
    model.compile(loss='mse', optimizer=optim, metrics = ['accuracy'])
    
    #Training and validating the model
    model.fit(
        x = [X_train.Game_Enc, X_train.User_Enc], 
        y = y_train.Rating, 
        batch_size = 128, 
        epochs = 100, #A high value can be used sicne early stopping is enabled
        verbose = 1, 
        validation_data = ([X_valid.Game_Enc, X_valid.User_Enc], y_valid.Rating), 
        callbacks = [early_stopping_cb, tensorboard_cb]
    )
    
    #Evaluating the model on the validation set using the optimal hyperparameters and saving the mse(loss)
    #The test set is not used in this case because the model has to be unbiased towards the test set when presenting the error
    loss, acc = model.evaluate([X_valid.Game_Enc, X_valid.User_Enc], y_valid.Rating)
    
    return {'loss': loss, 'status': STATUS_OK, 'model': model}



#data retrieval and processing function for hyperas parameter tuning
def data():
    """
    Note that the data has to be retrieved within this function even if it has retrieved 
    already at some other point in the script.

    Data loading function for hyperas parameter tuning. Provides the data that is passed into the model function.

    Arguments: None

    Returns:
    -> Trainset inputs (X_train)
    -> Trainset Outputs (y_train)
    -> Validation Inputs (X_valid)
    -> Validation Outputs (y_valid)

    """
    
    #Importing the train and validation sets from file on device
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

	#Hyperas Model optimization using the above defined functions
    #Returns the optimal parameter values (best_run) and the best model(best_model) from the defined search space
    best_run, best_model = optim.minimize(
    	model = model,
        data = data,
        algo = rand.suggest, #Search algorithm for traversing the search space
        max_evals = 5,
        trials = Trials(),
        eval_space = True #Shows the actual values chosen rather than the index of values chosen during tuning
    )

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    #Saving the best model so it can be imported in the notebook
    best_model.save("MLP_RecSys_Model.h5")


#End of hyperparameter tuning process
