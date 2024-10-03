'''
Author: Aditya Bhatt  7:53 PM 03/10/2024

Objective: Load the movielens 100k dataset and create a user history dictionary

TODO :

BUG:

'''



import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_movielens_100k(path='../Data/ml-100k/'):
    # Load ratings data
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(path + 'u.data', sep='\t', names=columns)
    # print(ratings.head())
    # print(ratings.shape)

    # Load movie data
    movie_columns = ['item_id', 'title', 'release_date', 'video_release_date', 
                     'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                     'Thriller', 'War', 'Western']
    movies = pd.read_csv(path + 'u.item', sep='|', names=movie_columns, encoding='latin-1')
    # print(movies.head())
    # print(movies.shape)
    # Merge ratings with movie titles
    df = pd.merge(ratings, movies[['item_id', 'title']], on='item_id')
    # print(df.head())
    # print(df.shape)
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Sort by user_id and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    # print(df.head())
    # Create user history
    user_history = df.groupby('user_id').apply(lambda x: x['item_id'].tolist()).to_dict()

    return df, user_history

# Load the data
df, user_history = load_movielens_100k()

print("Sample DataFrame (first 10 rows):")
print(df.head(10))
print("\nUser History (first 5 users):")
print(dict(list(user_history.items())[:5]))