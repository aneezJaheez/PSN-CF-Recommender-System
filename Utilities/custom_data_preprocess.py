from Utilities import data_preprocess as dp
import numpy as np
import pandas as pd

def split_dataset(df, train_size = 0.9):
    """Returns train and test sets in a 3-column format with columns as Users, Games, and Ratings respectively.
       The function creates the encoding for all the users and items, and then retrives the user(encoded), item(encoded),
       and ratings vectors.
       These vectors are then passed into the train_test_split_cf function which returns two arrays corresponding
       to the train and test array.
       Training examples present in both arrays are excluded from the test array.
       These arrays are then converted into a dataframe, melted down into a 3-column dataframe and the encodings are replaced with the original string values.
       
       Parameters:
           df : pandas dataframe, shape(, 3)
           Column names : ["User_Name", "Game_Name", "Rating"]
       
       Returns:
           trainset : pandas dataframe
           testset : pandas dataframe"""
    
    
    #The train_test_split_cf function defined above assumes that ratings with a value of 0 are empty cells.
    #However, this is not the case for this dataset. 
    #Convert the 0 ratings to -1 temporarily to differentiate between empty cells and zeros. 
    #This will be converted back to 0 after the empty cells are removed.
    df["Rating"].loc[df['Rating'] == 0] = -1
    
    #Retrieving all the unique game names
    game_names = df["Game_Name"].unique().tolist()
    #Creating encoding and reverse encoding
    #The encoded data will be used for the train-cv-test splits, and the reverse encodings will be used
    #to convert the data back to the original formate
    gamename_enc = {gamename : enc_value for enc_value, gamename in enumerate(game_names)}
    gamename_rev_enc = {gamename : enc_value for gamename, enc_value in enumerate(game_names)}
    
    #Performing the same steps for the users
    user_names = df["User_Name"].unique().tolist()
    username_enc = {username : enc_value for enc_value, username in enumerate(user_names)}
    username_rev_enc = {username : enc_value for username, enc_value in enumerate(user_names)}
    
    #Placing the encodings as separate columns in the dataset and mapping them so they match the other columns
    df["User_Enc"] = df["User_Name"].map(username_enc)
    df["Game_Enc"] = df["Game_Name"].map(gamename_enc)
    
    #Retrieving the vectors for the user and item encodings, and the ratings.
    #These vectors will be fed into the main train_test_split function
    u = np.array(df["User_Enc"])
    i = np.array(df["Game_Enc"])
    r = np.array(df["Rating"])
    
    trainset, testset = dp.train_test_split_cf(u, i, r, train_size=train_size)
    
    #After we have gotten the train and test sets, we use the reverse encodings the convert it back to the original format
    trainset = pd.DataFrame(trainset.toarray())
    testset = pd.DataFrame(testset.toarray())
    
    #Mapping the reverse encodings to the encodings so we have to original game names and usernames
    cols = pd.DataFrame(list(trainset.columns))
    cols.columns = ["Game_Encoding"]
    cols["Games"] = cols["Game_Encoding"].map(gamename_rev_enc)
    gamename_orig = list(cols["Games"])
    
    trainset.columns = gamename_orig
    testset.columns = gamename_orig
    
    usernames = list(username_enc.keys())
    trainset.insert(0, "User_Name", usernames)
    testset.insert(0, "User_Name", usernames)
    
    #Note that the data is in the form of a user item matrix, we will now convert it into a 3 column dataframe 
    #using the melt function and rename the columns
    trainset = pd.melt(trainset, id_vars = ["User_Name"])
    testset = pd.melt(testset, id_vars = ["User_Name"])
    
    trainset.columns = ["User_Name", "Game_Name", "Rating"]
    testset.columns = ["User_Name", "Game_Name", "Rating"]
    
    #Replacing the 0(represents empty entries) values with null so they can be removed from the dataframe
    trainset["Rating"] = trainset["Rating"].replace({0:np.nan})
    testset["Rating"] = testset["Rating"].replace({0:np.nan})
    
    #Retrieving the ratings we converted to -1 to differentiate from the empty entries
    trainset["Rating"].loc[trainset["Rating"] == -1] = 0
    testset["Rating"].loc[testset["Rating"] == -1] = 0
    
    trainset = trainset.dropna()
    testset = testset.dropna()
    
    #Note that the testset contains all the ratings from the trainset and then some extra ratings
    #Dropping the common ratings from the testset
    testset = testset[~testset.isin(trainset)].dropna()
    
    #Now we can return the final train and test datasets
    return trainset, testset