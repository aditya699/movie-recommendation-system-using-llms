# Movie Magic Recommender

## Overview

Movie Magic Recommender is an AI-powered movie recommendation system inspired by NVIDIA's LlamaRec approach. It combines collaborative filtering with large language models (LLMs) to provide personalized movie suggestions through an interactive web application.

![Movie App Screenshot 1](https://saibaba9758140479.blob.core.windows.net/testimages/movie_app_1.PNG)

## Features

- User-friendly web interface built with Streamlit
- Two-stage recommendation process: collaborative filtering for retrieval followed by LLM-based ranking
- Personalized recommendations based on user viewing history
- Interactive UI with movie posters and detailed information
- Integration with Anthropic's Claude API for advanced language processing

## How It Works

### 1. Retrieval Stage

The system employs collaborative filtering for the initial retrieval of candidate movies. This approach leverages user-item interactions across the entire user base, aiming to capture diverse and interesting recommendations.

### 2. Ranking Stage

The ranking stage utilizes Anthropic's Claude API to perform a nuanced ranking of the retrieved candidates. This allows for:

- Detailed analysis of user preferences
- Consideration of complex factors like themes, directors, and actors

### 3. User Interface

The project features a full-fledged web application:

- Intuitive web interface built with Streamlit
- Visual presentation of recommendations with movie posters
- Interactive elements for exploring recommended movies

![Movie App Screenshot 2](https://saibaba9758140479.blob.core.windows.net/testimages/movie_app_2.PNG)

## Inspiration from NVIDIA's LlamaRec

This project draws inspiration from NVIDIA's LlamaRec in several ways:

- Adoption of a two-stage recommendation process
- Use of large language models for ranking
- Focus on enhancing recommendation quality through advanced AI techniques

While LlamaRec emphasizes backend performance, this project explores the integration of these concepts into a user-facing application, balancing efficiency with a rich user experience.

## Getting Started

To set up and run the Movie Magic Recommender:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/movie-magic-recommender.git
   cd movie-magic-recommender
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the project root directory
   - Add your Anthropic API key:
     ```
     ANTHROPIC_API_KEY=your_api_key_here
     ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`)

## Contributors

- [Aditya Bhatt](https://github.com/aditya699)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This project explores the application of cutting-edge AI technology in creating a user-friendly, personalized movie recommendation system. By combining collaborative filtering with advanced language models, it aims to create an engaging experience for movie enthusiasts.