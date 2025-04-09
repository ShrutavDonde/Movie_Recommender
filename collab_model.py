from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
import pandas as pd

class CollaborativeRecommender:
    def __init__(self):
        self.model = SVD()

    def fit(self, ratings_df):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)

    def predict(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id).est