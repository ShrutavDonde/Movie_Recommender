from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentRecommender:
    def __init__(self, metadata_column='metadata'):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.metadata_column = metadata_column
        self.similarity_matrix = None

    def fit(self, df):
        tfidf_matrix = self.vectorizer.fit_transform(df[self.metadata_column])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        self.movie_indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    def get_similar_movies(self, title, top_n=10):
        idx = self.movie_indices.get(title)
        if idx is None:
            return []
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        return sim_scores