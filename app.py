


#importing the dependencies
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""data collection and pre processing"""

#loading a data from the csv files to apandas dataframe
movies_data = pd.read_csv('movies.csv')

#printing the first five line or rows of the data
movies_data.head()

#total number of rows and columns are present in the data
movies_data.shape

#selecting the relevant features of the recommendation
selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)

#replacing the null values with nullstring
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

#combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

print(combined_features)

#converting the text data to feature vector

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

print(feature_vectors)

"""cosine similarity"""

#getting the similarity score using cosine similarity
similarity = cosine_similarity(feature_vectors)

print(similarity)

print(similarity.shape)

#getting the movies name from the user
movie_name = input('enter your favourite movie name : ')

#creating a  list with all the movies names given in the dataset
list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)

#finding the close match for the movies name given by the user
find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

#finding the index of the movies with the title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)

#getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)

len(similarity_score)

#sorting the movies based on their similarity score
similar_movies = sorted(similarity_score,key = lambda x:x[1],reverse = True)
print(similar_movies)

#print the name of the similar movies based on the index

print('Movies suggested for you : \n')
i = 1
for movies in similar_movies:
  index = movies[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<31):
    print(i,'.',title_from_index)
    i+=1

"""movies recommendation system

"""

movie_name = input('enter your favourite movie name : ')
list_of_all_titles = movies_data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)
close_match = find_close_match[0]
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
similar_movies = sorted(similarity_score,key = lambda x:x[1],reverse = True)

print('Movies suggested for you : \n')

i = 1

for movies in similar_movies:
  index = movies[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<31):
    print(i,'.',title_from_index)
    i+=1
