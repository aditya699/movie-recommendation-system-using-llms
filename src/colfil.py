'''
Author: Aditya Bhatt 10:11 AM 03/10/2024

NOTE :
1.Collaborative filtering is a technique used in recommender systems to make predictions about a user's interests by collecting preferences from many users.
2.The idea is that if Person A has similar opinions to Person B on an issue, A is more likely to have B's opinion on a different issue compared to that of a randomly chosen person.
3.There are two main types of collaborative filtering:

User-based: Finds users with similar tastes and recommends items they liked.
Item-based: Finds items similar to those the user has liked in the past

4.Retrieval Stage: This is where collaborative filtering comes into play. The paper uses LRURec, 
which is a sequential recommender, for the retrieval stage. 
While LRURec isn't a traditional collaborative filtering method, it does use patterns in user behavior (which is a form of implicit collaboration) to make recommendations.

Ranking Stage: This uses a large language model (LLM) to rank the retrieved items.

'''
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_movielens_100k


def create_user_item_matrix(df):
    """
    Create a user-item matrix from the dataframe.
    
    Args:
    df (pandas.DataFrame): Dataframe with columns 'user_id', 'item_id', and 'rating'
    
    Returns:
    numpy.ndarray: User-item matrix
    dict: Mapping of matrix index to user_id
    dict: Mapping of matrix index to item_id
    """
    # Create pivot table
    user_item_df = df.pivot(index='user_id', columns='item_id', values='rating')
    print(user_item_df.head())
    
    # Fill NaN values with 0
    user_item_df = user_item_df.fillna(0)
    print(user_item_df.head())
    
    # Create mappings
    user_to_index = {user: i for i, user in enumerate(user_item_df.index)}
    item_to_index = {item: i for i, item in enumerate(user_item_df.columns)}
    print(user_to_index)
    print(item_to_index)
    
    return user_item_df.values, user_to_index, item_to_index

def calculate_item_similarities(user_item_matrix):
    """
    Calculate item-item similarities using cosine similarity.
    
    Args:
    user_item_matrix (numpy.ndarray): User-item matrix
    
    Returns:
    numpy.ndarray: Item-item similarity matrix
    """
    print(user_item_matrix)
    print(user_item_matrix.T)
    return cosine_similarity(user_item_matrix.T)

# Load and prepare data
df, user_history = load_movielens_100k()
user_item_matrix, user_to_index, item_to_index = create_user_item_matrix(df)

# Calculate item similarities
item_similarities = calculate_item_similarities(user_item_matrix)
print(item_similarities)
print("Shape of item similarity matrix:", item_similarities.shape)
print("Sample similarity between item 0 and item 1:", item_similarities[0][1])