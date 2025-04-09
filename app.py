import streamlit as st # type: ignore
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
movies = pd.read_csv('movies.csv')

# Preprocess genres
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Title to index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommend function
def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'poster_url']].values.tolist()

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get similar recommendations!")

selected_movie = st.selectbox("Choose a movie:", movies['title'].tolist())

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)
    if recommendations:
        st.subheader("🎥 You may also like:")
        for title, poster_url in recommendations:
            st.markdown(f"**{title}**")
            st.image(poster_url, width=200)
    else:
        st.write("No recommendations found. Try another movie.")
