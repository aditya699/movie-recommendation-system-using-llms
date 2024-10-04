import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Small example user-item matrix
user_item_matrix = np.array([
    [5, 3, 0, 1],  # User 1
    [4, 0, 0, 1],  # User 2
    [1, 1, 0, 5],  # User 3
    [1, 0, 0, 4],  # User 4
    [0, 1, 5, 4],  # User 5
])

# Calculate item similarities
item_similarities = cosine_similarity(user_item_matrix.T)

print("Item Similarities Matrix:")
print(item_similarities)

# Let's say we're recommending for User 2
user_history = [0, 3]  # User 2 has interacted with items 0 and 3

# Select similarities to items in user's history
relevant_similarities = item_similarities[:, user_history]
print("\nRelevant Similarities:")
print(relevant_similarities)

# Calculate average similarities
avg_similarities = np.mean(relevant_similarities, axis=1)
print("\nAverage Similarities:")
print(avg_similarities)

# Get recommendations (excluding items in history)
recommendations = np.argsort(avg_similarities)[::-1]
recommendations = [item for item in recommendations if item not in user_history]
print("\nRecommended Items (in order):")
print(recommendations)