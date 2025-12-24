import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Content-based recommendation using TF-IDF & cosine similarity")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        df[feature] = df[feature].fillna('')
    return df

df = load_data()

# -------------------------------
# Feature Engineering
# -------------------------------
@st.cache_resource
def build_similarity(dataframe):
    combined_features = (
        dataframe['genres'] + ' ' +
        dataframe['keywords'] + ' ' +
        dataframe['tagline'] + ' ' +
        dataframe['cast'] + ' ' +
        dataframe['director']
    )

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    similarity = cosine_similarity(feature_vectors)
    return similarity

similarity = build_similarity(df)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_movies(movie_name):
    list_of_all_titles = df['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return []

    close_match = find_close_match[0]
    index_of_the_movie = df[df.title == close_match]['index'].values[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(
        similarity_score,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:11], start=1):
        index = movie[0]
        title_from_index = df[df.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)

    return recommended_movies, close_match

# -------------------------------
# UI Section
# -------------------------------
movie_name = st.text_input("🎥 Enter your favourite movie name")

if st.button("🔍 Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        results = recommend_movies(movie_name)

        if not results:
            st.error("No similar movie found. Try another name.")
        else:
            recommendations, matched_movie = results
            st.success(f"Showing recommendations for: **{matched_movie}**")
            st.subheader("🍿 Recommended Movies")

            for i, movie in enumerate(recommendations, start=1):
                st.write(f"{i}. {movie}")
