from utils import load_ratings, load_movies
from content_model import ContentRecommender
from collab_model import CollaborativeRecommender
from stacked_model import StackedRecommender
import pandas as pd

# Load data
ratings = load_ratings('data/ratings.csv')
movies = load_movies('data/movie_subset_with_metadata.csv')

# Init models
content_rec = ContentRecommender(metadata_column='metadata')
collab_rec = CollaborativeRecommender()
stacked_rec = StackedRecommender()

# Train content model
movies['metadata'] = movies[['genres', 'cast', 'director']].fillna('').agg(' '.join, axis=1)
content_rec.fit(movies)

# Train collaborative model
collab_rec.fit(ratings)

# Create features for stacking
stacked_df = ratings.copy()
stacked_df['collab_pred'] = ratings.apply(lambda row: collab_rec.predict(row['userId'], row['movieId']), axis=1)
stacked_df = stacked_df.merge(movies[['movieId', 'title']], on='movieId', how='left')
stacked_df['content_pred'] = stacked_df['title'].apply(lambda x: content_rec.get_similar_movies(x, top_n=1)[0][1] if content_rec.get_similar_movies(x, top_n=1) else 0)

X = stacked_df[['collab_pred', 'content_pred']]
y = stacked_df['rating']

# Train stacked model
stacked_rec.fit(X, y)

# Predict
stacked_df['stacked_pred'] = stacked_rec.predict(X)
print(stacked_df[['userId', 'movieId', 'rating', 'stacked_pred']].head())