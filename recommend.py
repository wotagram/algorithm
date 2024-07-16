import torch
import pandas as pd
from src.data_loader import load_data

posts_dataframe, ratings_dataframe = load_data(
    'data/posts.csv', 'data/ratings.csv')


def get_recommendations(model, user_id, posts_dataframe, ratings_dataframe, top_n=10):
    # Filter posts that the user has not rated
    user_rated_posts = ratings_dataframe[ratings_dataframe['user_id']
                                         == user_id]['post_id'].tolist()
    all_posts = posts_dataframe['id'].tolist()

    posts_to_predict = list(set(all_posts) - set(user_rated_posts))

    # Prepare data for prediction
    user_ids = [user_id] * len(posts_to_predict)

    # Create tensor for prediction
    data = torch.tensor(
        list(zip(user_ids, posts_to_predict)), dtype=torch.long)

    # Predict ratings for user-item pairs
    predictions = model(data)

    # Combine predictions with post IDs
    post_predictions = list(
        zip(posts_to_predict, predictions.detach().numpy()))

    # Sort predictions by rating (descending order)
    post_predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top N recommended posts
    top_recommendations = post_predictions[:top_n]

    # Fetch content of recommended posts from posts_dataframe
    recommended_posts = [(post_id, posts_dataframe.loc[posts_dataframe['id']
                          == post_id]['content'].iloc[0]) for post_id, _ in top_recommendations]

    return recommended_posts
