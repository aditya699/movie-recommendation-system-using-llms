# src/app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import os
from dotenv import load_dotenv
import random
# Set page config at the very beginning
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Custom CSS to make the app more visually appealing
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(to right, #141E30, #243B55);
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #FF4B2B;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #FF416C;
        transform: scale(1.05);
    }
    .movie-title {
        font-size: 18px;
        color: #FFD700;
        margin-bottom: 5px;
    }
    .movie-rank {
        font-size: 24px;
        font-weight: bold;
        color: #FF4B2B;
    }
    .sidebar .stSelectbox {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)
# Load environment variables
load_dotenv()

# Set up Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load data
@st.cache_data
def load_movielens_100k(path='Data/ml-100k/'):
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(path + 'u.data', sep='\t', names=columns)
    
    movie_columns = ['item_id', 'title', 'release_date', 'video_release_date', 
                     'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                     'Thriller', 'War', 'Western']
    movies = pd.read_csv(path + 'u.item', sep='|', names=movie_columns, encoding='latin-1')
    
    df = pd.merge(ratings, movies[['item_id', 'title']], on='item_id')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(['user_id', 'timestamp'])
    
    user_history = df.groupby('user_id').apply(lambda x: x['item_id'].tolist()).to_dict()
    
    return df, user_history

# Create user-item matrix
@st.cache_data
def create_user_item_matrix(df):
    user_item_df = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    user_to_index = {user: i for i, user in enumerate(user_item_df.index)}
    item_to_index = {item: i for i, item in enumerate(user_item_df.columns)}
    return user_item_df.values, user_to_index, item_to_index

# Calculate item similarities
@st.cache_data
def calculate_item_similarities(user_item_matrix):
    return cosine_similarity(user_item_matrix.T)

# Retrieve candidates
def retrieve(user_id, item_history, user_item_matrix, item_similarities, user_to_index, item_to_index, n_candidates=20):
    if user_id not in user_to_index:
        return np.random.choice(list(item_to_index.keys()), size=n_candidates, replace=False)
    
    user_index = user_to_index[user_id]
    history_indices = [item_to_index[item] for item in item_history if item in item_to_index]
    avg_similarities = np.mean(item_similarities[:, history_indices], axis=1)
    mask = np.ones(len(avg_similarities), dtype=bool)
    mask[history_indices] = False
    masked_similarities = avg_similarities[mask]
    candidate_indices = np.argsort(masked_similarities)[::-1][:n_candidates]
    original_indices = np.arange(len(avg_similarities))[mask][candidate_indices]
    index_to_item = {v: k for k, v in item_to_index.items()}
    candidates = [index_to_item[idx] for idx in original_indices]
    return candidates

# Get titles
def get_titles(item_ids, item_id_to_title):
    return [item_id_to_title.get(item_id, f"Unknown Movie {item_id}") for item_id in item_ids]

# Create ranking prompt
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

# Parse ranking response
def parse_strict_ranking_response(response_content, candidate_titles):
    ranking_letters = response_content.strip().split(',')
    ranked_movies = []
    for rank, letter in enumerate(ranking_letters):
        index = ord(letter.strip()) - ord('A')
        if 0 <= index < len(candidate_titles):
            ranked_movies.append((rank + 1, candidate_titles[index]))
    return ranked_movies

# Load data and prepare matrices
df, user_history = load_movielens_100k()
user_item_matrix, user_to_index, item_to_index = create_user_item_matrix(df)
item_similarities = calculate_item_similarities(user_item_matrix)
item_id_to_title = dict(zip(df['item_id'], df['title']))

# Load data and prepare matrices
df, user_history = load_movielens_100k()
user_item_matrix, user_to_index, item_to_index = create_user_item_matrix(df)
item_similarities = calculate_item_similarities(user_item_matrix)
item_id_to_title = dict(zip(df['item_id'], df['title']))

# Streamlit app
st.title("🎬 Movie Magic Recommender")
st.markdown("Discover your next favorite film with the power of AI!")

# Sidebar
st.sidebar.image("https://img.freepik.com/free-vector/cinema-realistic-poster-with-illuminated-bucket-popcorn-drink-3d-glasses-reel-tickets-blue-background-with-tapes-vector-illustration_1284-77070.jpg", use_column_width=True)
st.sidebar.header("🎭 User Selection")

# User input for user ID
user_input = st.sidebar.text_input("Enter a user ID:", "")

# Validate user input
if user_input:
    try:
        selected_user = int(user_input)
        if selected_user not in user_history:
            st.sidebar.error(f"User ID {selected_user} not found. Please enter a valid user ID.")
            selected_user = None
        else:
            st.sidebar.success(f"User ID {selected_user} selected!")
    except ValueError:
        st.sidebar.error("Please enter a valid numeric user ID.")
        selected_user = None
else:
    selected_user = None

# Display some information about available user IDs
st.sidebar.info(f"Available user IDs range from {min(user_history.keys())} to {max(user_history.keys())}.")

# Main content
if selected_user:
    st.subheader(f"🍿 User {selected_user}'s Movie History")
    user_movies = get_titles(user_history[selected_user], item_id_to_title)
    
    # Display user's movie history in a more appealing way
    cols = st.columns(3)
    for i, movie in enumerate(user_movies):
        with cols[i % 3]:
            st.markdown(f"<div class='movie-title'>🎥 {movie}</div>", unsafe_allow_html=True)

    if st.button("🚀 Get Magical Recommendations"):
        with st.spinner("🧙‍♂️ Our AI wizard is conjuring up recommendations..."):
            candidates = retrieve(selected_user, user_history[selected_user], user_item_matrix, item_similarities, user_to_index, item_to_index)
            candidate_titles = get_titles(candidates, item_id_to_title)
            
            prompt = create_strict_ranking_prompt(user_movies, candidate_titles)
            
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            ranked_movies = parse_strict_ranking_response(message.content[0].text, candidate_titles)
            
            st.subheader("🌟 Your Personalized Movie Magic")
            
            # Display recommendations in a more visually appealing way
            for rank, movie in ranked_movies:
                with st.expander(f"{rank}. {movie}"):
                    st.markdown(f"<div class='movie-rank'>#{rank}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='movie-title'>{movie}</div>", unsafe_allow_html=True)
                

        # Add a fun fact or movie quote
        movie_quotes = [
            "Here's looking at you, kid. - Casablanca",
            "I'm going to make him an offer he can't refuse. - The Godfather",
            "May the Force be with you. - Star Wars",
            "You talkin' to me? - Taxi Driver",
            "E.T. phone home. - E.T. the Extra-Terrestrial"
        ]
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Movie Magic Quote:**\n\n*{random.choice(movie_quotes)}*")

st.sidebar.markdown("---")
st.sidebar.info("This app uses collaborative filtering and AI to recommend movies based on user history.")
st.sidebar.text("Made with ❤️ by Your Favorite AI")

# Add a footer
st.markdown("---")
st.markdown("Created with 🎬 by Aditya Bhatt | © 2024")