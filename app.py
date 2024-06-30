from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

app = FastAPI()

# Load your movies data
movies_data = pd.read_csv(r'C:\Users\user\Documents\TAB\movies.csv')  # Update this path

# Selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replacing the null values with an empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining the selected features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Creating the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Generating the feature vectors
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculating the cosine similarity
similarity = cosine_similarity(feature_vectors)

class MovieRequest(BaseModel):
    movie_name: str

@app.post("/recommendations/")
def get_recommendations(request: MovieRequest):
    movie_name = request.movie_name
    list_of_all_titles = movies_data['title'].tolist()

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        raise HTTPException(status_code=404, detail="Movie not found")

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie in sorted_similar_movies[:30]:  # Limit to top 30 recommendations
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommendations.append(title_from_index)

    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
