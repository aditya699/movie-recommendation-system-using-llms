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
import anthropic

import os


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

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

def retrieve(user_id, item_history, user_item_matrix, item_similarities, user_to_index, item_to_index, n_candidates=20):
    if user_id not in user_to_index:
        print(f"User {user_id} not in training data. Returning random items.")
        return np.random.choice(list(item_to_index.keys()), size=n_candidates, replace=False)

    user_index = user_to_index[user_id]
    user_vector = user_item_matrix[user_index]
    
    # Get indices of items in the user's history
    history_indices = [item_to_index[item] for item in item_history if item in item_to_index]
    print("User history indices:", history_indices)
    
    # Calculate the average similarity to items in the user's history
    avg_similarities = np.mean(item_similarities[:, history_indices], axis=1)
    
    # Create a mask to exclude items in the user's history
    mask = np.ones(len(avg_similarities), dtype=bool)
    mask[history_indices] = False
    print("Number of items excluded by mask:", np.sum(~mask))
    
    # Sort items by similarity and get top N candidates, excluding items in history
    masked_similarities = avg_similarities[mask]
    candidate_indices = np.argsort(masked_similarities)[::-1][:n_candidates]
    
    # Map the masked indices back to the original indices
    original_indices = np.arange(len(avg_similarities))[mask][candidate_indices]
    
    # Convert indices back to item IDs
    index_to_item = {v: k for k, v in item_to_index.items()}
    candidates = [index_to_item[idx] for idx in original_indices]
    
    print(f"Retrieved {len(candidates)} candidates for user {user_id}")
    print("Sample candidates:", candidates[:5])
    
    # Check if any history items are in the candidates
    intersection = set(item_history) & set(candidates)
    if intersection:
        print("WARNING: Some history items are still in candidates:", intersection)
    else:
        print("Success: No history items in candidates")
    
    return candidates
# Load and prepare data
df, user_history = load_movielens_100k()
item_id_to_title = dict(zip(df['item_id'], df['title']))
# Function to get titles
def get_titles(item_ids):
    return [item_id_to_title.get(item_id, f"Unknown Movie {item_id}") for item_id in item_ids]



user_item_matrix, user_to_index, item_to_index = create_user_item_matrix(df)

def create_strict_ranking_prompt(history_titles, candidate_titles):
    history_str = "\n".join(f"- {title}" for title in history_titles)
    candidates_str = "\n".join(f"({chr(65+i)}) {title}" for i, title in enumerate(candidate_titles))
    
    prompt = f"""Given a user's movie viewing history, rank all the candidate movies from most to least suitable. Consider factors such as genre, themes, directors, and actors when making your rankings.

User's viewing history:
{history_str}

Candidate movies:
{candidates_str}

IMPORTANT: Provide your rankings ONLY as a comma-separated list of letters corresponding to the movies, from most to least suitable. Do not include any other text, explanations, or spaces.

Example correct output:
C,A,F,B,D,E,G,H,I,J,K,L,M,N,O,P,Q,R,S,T

Your ranking (letters only, comma-separated, no spaces):
"""
    
    return prompt

def parse_strict_ranking_response(response_content, candidate_titles):
    # Extract the ranking letters from the response
    ranking_letters = response_content[0].text.strip().split(',')
    
    # Create a list of (rank, movie title) tuples
    ranked_movies = []
    for rank, letter in enumerate(ranking_letters):
        index = ord(letter.strip()) - ord('A')
        if 0 <= index < len(candidate_titles):
            ranked_movies.append((rank + 1, candidate_titles[index]))
    
    return ranked_movies
# Calculate item similarities
item_similarities = calculate_item_similarities(user_item_matrix)
print(item_similarities)
print("Shape of item similarity matrix:", item_similarities.shape)
print("Sample similarity between item 0 and item 1:", item_similarities[0][1])
# Step 5: Use the retrieve function to get recommendations
test_user_id = df['user_id'].iloc[0]

test_history = user_history[test_user_id] 

print(f"\nTesting retrieval for user {test_user_id}")
print("User history:", test_history)

candidates = retrieve(test_user_id, test_history, user_item_matrix, item_similarities, user_to_index, item_to_index)
print("Retrieved candidates:", candidates)

# Get titles for user history and retrieved candidates
history_titles = get_titles(test_history)
print(history_titles)
candidate_titles = get_titles(candidates)
print(candidate_titles)

# Create the strict ranking prompt
prompt = create_strict_ranking_prompt(history_titles, candidate_titles)
print("Prompt sent to LLM:")
print(prompt)

# Send the prompt to Claude API
message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print("\nLLM Response:")
print(message.content)

# Parse the strict ranking response
ranked_movies = parse_strict_ranking_response(message.content, candidate_titles)

# Display the results
print(f"\nRanked recommendations for user {test_user_id}:")
for rank, movie in ranked_movies:
    print(f"{rank}. {movie}")

print("\nUser's viewing history:")
for title in history_titles:
    print(f"- {title}")