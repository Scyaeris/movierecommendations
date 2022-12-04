#Import sklearn (KNN)
from sklearn.neighbors import NearestNeighbors

#Import data visualization & matrix
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Import basic tools
import numpy as np
import pandas as pd

#Import FuzzyWuzzy
from fuzzywuzzy import fuzz

#Import movies and ratings .csv files
ratings = pd.read_csv('movies/ratings.csv')
movies = pd.read_csv('movies/movies.csv')

#The "dropna = True" parameter tells the code to ignore any entries with missing values. 
#So the code will count the number of unique users and movies in the ratings data, excluding any entries with missing values.
unique_user = ratings.userId.nunique(dropna = True)
unique_movie = ratings.movieId.nunique(dropna = True)

# In order to create item user matrixes, we need to check how many ratings exist within our dataframes/and/or how many ratings are absent
total_ratings = unique_user * unique_movie
rating_present = ratings.shape[0]

ratings_not_provided = total_ratings - rating_present 

"""Data Exploration"""

#Gives information about which rating (on scale of 0 to 5) is more frequent
rating_cnt = pd.DataFrame(ratings.groupby('rating').size(),columns=['count'])
#This rating count does not contain any ratings of value 0

rating_cnt = rating_cnt.append(pd.DataFrame({'count' : ratings_not_provided}, index = [0])).sort_index()

#Since there are more "0 ratings" as compared to other ratings, use the log value
#Just for the visualization
rating_cnt['log_count'] = np.log(rating_cnt['count'])
rating_cnt = rating_cnt.drop(rating_cnt.index[0])

movie_freq = pd.DataFrame(ratings.groupby('movieId').size(), columns = ['count'])


threshold_rating_freq = 50
#Take out the movie IDs for which each movie is rated more than threshold value 
#Then, keep only the movies in our original ratings dataframe

popular_movies_id = list(set(movie_freq.query('count >= @threshold_rating_freq').index))

#Ratings dataframe after dropping "non-popular" movies
ratings_with_popular_movies = ratings[ratings.movieId.isin(popular_movies_id)]

"""User Analysis"""

#How many times each user (userId) rates movies
user_cnt = pd.DataFrame(ratings.groupby('userId').size(),columns = ['count'])
uc_copy = user_cnt

"""Rating Frequency by User follows tail trend: generally there are just fewer users who are interseted in rating movies

There are few users that only avidly rate movies
"""

# Lets find users who have more than 30 ratings
threshold_val = 30
active_user = list(set(user_cnt.query('count >= @threshold_val').index))

#update your ratings_with_popular_movies
ratings_with_popular_movies_with_active_user = ratings_with_popular_movies[ratings_with_popular_movies.userId.isin(active_user)]

"""Sparsity reduced by removing unpopular movies and inactive users who only reviewed a select amount of movies

#Building KNN Model

We have to reshape/prepare our dataset into a format which can be given as parameter to the KNN function. We will pivot our final dataset into a ITEM-USER matrix and fill empty cells with with the vlaue 0 because the KNN model calculates distances between two points.
"""

#final_ratings is a dataframe with the ratings of the movies that are popular and the users that are active
#item_user_mat is a matrix with the movies as rows and the users as columns. The values are the ratings.
#If there is no rating, the value is 0.
final_ratings = ratings_with_popular_movies_with_active_user
item_user_mat = final_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating').fillna(0)

#Creates a dictionary called movie_to_index, with the movie titles as keys and the numerical index as values. 
#It does this by enumerating the movie titles from the list of movies and setting the index to the movieId values from the item_user_mat.
movie_to_index = {
    movie:i for i, movie in enumerate(list(movies.set_index('movieId').loc[item_user_mat.index].title))
}

#Creates a sparse matrix from the item_user_mat matrix. 
#A sparse matrix is a data structure that only stores the nonzero elements of a matrix, thus saving memory. 
#The csr_matrix() function is used to create a Compressed Sparse Row matrix from the values of the item_user_mat matrix.
item_user_mat_sparse = csr_matrix(item_user_mat.values)

"""When a movie name is given as input we need to find that movie in our dataset. If it is not present then we can not recommend anything. 

"""

#"find" function will find the closest matching string from a given set of strings. 
#The input_str is a string the user is searching for, and mapper is a set of strings. 
#The code searches through the set of strings and uses fuzz.ratio to compare the input_str to the strings in mapper. 
#If the ratio is greater than or equal to 50, it is added to the match_movie list. 
#The list is then sorted by the ratio in decreasing order and the index of the highest ratio is returned. 
#If no match is found, -1 is returned.

def find(input_str, mapper):
    #match_movie is list of tuples, which have 3 values (movie_name, index, fuzz_ratio)
    match_movie = []
    for movie, ind in mapper.items():
        current_ratio = fuzz.ratio(movie.lower(), input_str.lower())
        if (current_ratio >= 50):
            match_movie.append((movie,ind,current_ratio))
     
    # Sort the list, match_movie, in order of fuzz_ratio
    match_movie = sorted(match_movie, key = lambda x:x[2])[::-1]
    
    if len(match_movie) == 0:
        #print("Movie string not found in dataset \n")
        return -1
    return match_movie[0][1]

#Define the model
#The metric used is 'cosine', which is a measure of similarity between two vectors. 
#The algorithm used is 'brute', which is a brute force algorithm for computing the nearest neighbors of a given point. 
#The n_neighbors parameter specifies the number of nearest neighbors to return. 
#The n_jobs parameter specifies the number of CPU cores to use for computing the model, in this case it is set to -1 which indicates that all available cores should be used.
recommendation_model = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 20, n_jobs = -1)

# Create a function which takes a movie name and make recommedation for it
def recommend(input_str, data, model, mapper, n_recommendation):
    # print("Processing....\n")
    # print("=====================================================")
    model.fit(data)
    
    # The mapper is a dictionary that maps each movie to its index in the dataset.
    index = find(input_str, mapper)
    
    if index == -1 :
        # print("Movie string not found in dataset \n")
        return 
    if n_recommendation == 0:
      #print("You can not enter 0 as a valid number of returned recomendations.")
      return

    index_list = model.kneighbors(data[index], n_neighbors = n_recommendation + 1, return_distance = False)
    # Create mapper index to title  
    index_to_movie = {
        ind:movie for movie,ind in mapper.items()
    }
    results = []
    #print("A viewer who watches",input_str,"should also watch the following movies: ")
    for i in range(1,index_list.shape[1]):
        #print(index_to_movie[index_list[0][i]])
        results.append(index_to_movie[index_list[0][i]])
    
    return results

def getCSR():
    return item_user_mat_sparse

def getModel():
    return recommendation_model

def getMovieToIndex():
    return movie_to_index

#test_movie, rec_movie = recommend(m, item_user_mat_sparse, recommendation_model, movie_to_index, int(n))

"""Observations (for the slides): 
*   Model recommends movies which are close in release years
*   If you calculate the cosine distance between the input movie and the recommended movie, this distance is very small due a large number of cells in our movie_user_mat dataframe being filled with 0. 
*   We have almost 92% sparsity in our final item_user_mat.
*   Since we have removed unpopular movies from our dataset, some movies will never be recommended to users
*   Popularity bias: This model only recommends movies which are tagged as "popluar" movies. Movies towards the "tail" of the curve (graphs above) are not recommended
"""