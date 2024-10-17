import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride Part II'],
    'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy']
}
movies = pd.DataFrame(movie_data)

rating_data = {
    'user_id': [1, 2, 3, 4, 5],
    'movie_id': [1, 2, 3, 4, 5],
    'rating': [4, 5, 2, 3, 4]
}
ratings = pd.DataFrame(rating_data)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_recommendations(movie_title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

print("Content-based recommendations for 'Toy Story':")
print(get_content_recommendations('Toy Story'))

user_id = 1
movie_id = 2
prediction = algo.predict(user_id, movie_id)
print(f"Predicted rating for User {user_id} on Movie {movie_id}: {prediction.est}")

def hybrid_recommendation(user_id, movie_title):
    content_recs = get_content_recommendations(movie_title)
    cf_predictions = []
    for movie in content_recs:
        movie_idx = movies[movies['title'] == movie]['movie_id'].values[0]
        cf_pred = algo.predict(user_id, movie_idx)
        cf_predictions.append((movie, cf_pred.est))
    cf_predictions = sorted(cf_predictions, key=lambda x: x[1], reverse=True)
    return cf_predictions

print("\nHybrid recommendations for User 1 based on 'Toy Story':")
print(hybrid_recommendation(1, 'Toy Story'))
