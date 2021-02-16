#!/usr/bin/env python
# coding: utf-8

# ## 1. Import libraries 

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer

import warnings; warnings.simplefilter('ignore')

print('Libraries imported....')


# ## 2. Load Dataset

# This dataset is MovieLens dataset.
# 
# This contains metadata for around 45,000 movies listed in the Full MovieLens dataset.Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
# 
# keywords_dataset.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
# 
# credits_dataset.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
# 
# tags_dataset.csv: The file that contains the TMDB and IMDB IDs of 9000 movies featured in the Full MovieLens dataset.
# 
# ratings_dataset.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.
# 

# In[2]:


metadata_movie = pd.read_csv('C:/Users/sameer.sakkhari/Desktop/Manasa/CS698/dataset_project/movies_metadata.csv')
ratings_dataset = pd.read_csv('C:/Users/sameer.sakkhari/Desktop/Manasa/CS698/dataset_project/ratings_dataset.csv')
credits_dataset = pd.read_csv('C:/Users/sameer.sakkhari/Desktop/Manasa/CS698/dataset_project/credits_dataset.csv')
keywords_dataset = pd.read_csv('C:/Users/sameer.sakkhari/Desktop/Manasa/CS698/dataset_project/keywords_dataset.csv')
tags_dataset = pd.read_csv('C:/Users/sameer.sakkhari/Desktop/Manasa/CS698/dataset_project/tags_dataset.csv')


# ## 3. Understand Dataset 

# In[3]:


metadata_movie.head()


# In[4]:


metadata_movie.columns


# 
# Features
# 
# - adult: Indicates if the movie is X-Rated or Adult.
# - belongs_to_collection: A stringified dictionary that gives information on the movie series the particular film belongs to.
# - budget: The budget of the movie in dollars.
# - genres: A stringified list of dictionaries that list out all the genres associated with the movie.
# - homepage: The Official Homepage of the move.
# - id: The ID of the movie.
# - imdb_id: The IMDB ID of the movie.
# - original_language: The language in which the movie was originally shot in.
# - original_title: The original title of the movie.
# - overview: A brief blurb of the movie.
# - popularity: The Popularity Score assigned by TMDB.
# - poster_path: The URL of the poster image.
# - production_companies: A stringified list of production companies involved with the making of the movie.
# - production_countries: A stringified list of countries where the movie was shot/produced in.
# - release_date: Theatrical Release Date of the movie.
# - revenue: The total revenue of the movie in dollars.
# - runtime: The runtime of the movie in minutes.
# - spoken_languages: A stringified list of spoken languages in the film.
# - status: The status of the movie (Released, To Be Released, Announced, etc.)
# - tagline: The tagline of the movie.
# - title: The Official Title of the movie.
# - video: Indicates if there is a video present of the movie with TMDB.
# - vote_average: The average rating of the movie.
# - vote_count: The number of votes by users, as counted by TMDB.

# In[5]:


metadata_movie.dtypes


# In[6]:


metadata_movie.shape


# In[7]:


metadata_movie.info()


# In[8]:


ratings_dataset.head()


# In[9]:


ratings_dataset.describe()


# In[10]:


ratings_dataset.columns


# - userId: It is id for User
# - movieId: It is TMDb movie id.
# - rating: Rating given for the particular movie by specific user
# - timestamp: Time stamp when rating has been given by user

# In[11]:


ratings_dataset.shape


# In[12]:


ratings_dataset.info()


# In[13]:


credits_dataset.head()


# In[14]:


credits_dataset.describe()


# In[15]:


credits_dataset.dtypes


# - cast: Information about casting. Name of actor, gender and it's character name in movie
# - crew: Information about crew members. Like who directed the movie, editor of the movie and so on.
# - id: It's movie ID given by TMDb

# In[16]:


credits_dataset.shape


# In[17]:


keywords_dataset.info()


# In[18]:


keywords_dataset.head()


# In[19]:


keywords_dataset.describe()


# In[20]:


keywords_dataset.dtypes


# In[21]:


keywords_dataset.shape


# In[22]:


keywords_dataset.info()


# In[23]:


keywords_dataset.columns


# - id: It's movie ID given by TMDb
# - Keywords: Tags/keywords for the movie. It list of tags/keywords

# In[24]:


tags_dataset.head()


# In[25]:


tags_dataset.describe()


# In[26]:


tags_dataset.dtypes


# In[27]:


tags_dataset.shape


# In[28]:


tags_dataset.info()


# - movieId: It's serial number for movie
# - imdbId: Movie id given on IMDb platform
# - tmdbId: Movie id given on TMDb platform

# ## 4. Build recommendation system 

# ### 4.1 Content based recommendation system 

# In[29]:


tags_dataset = tags_dataset.loc[tags_dataset['tmdbId'].notnull()]['tmdbId'].astype('int')
tags_dataset


# In[30]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[31]:


metadata_movie['id'] = metadata_movie['id'].apply(convert_int)
metadata_movie[metadata_movie['id'].isnull()]


# In[32]:


metadata_movie = metadata_movie.drop([19730, 29503, 35587])


# In[33]:


metadata_movie['id'] = metadata_movie['id'].astype('int')


# In[34]:


user_input = metadata_movie[metadata_movie['id'].isin(tags_dataset)]
user_input.shape


# In[35]:


user_input['overview'] = user_input['overview'].fillna('')
user_input['tagline'] = user_input['tagline'].fillna('')
user_input['description'] = user_input['overview'] + user_input['tagline']


# In[36]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0001, stop_words='english')
tfidf_matrix = tf.fit_transform(user_input['description'])


# In[37]:


tfidf_matrix.shape


# Instead of using dot product to obtains weights(cosine similarity score) in the user profile, I have used sklearn's linear_kernel as its faster.

# In[38]:


user_profile = linear_kernel(tfidf_matrix, tfidf_matrix)
user_profile


# In[39]:


user_input = user_input.reset_index()
titles = user_input['title']
indices = pd.Series(user_input.index, index=user_input['title'])


# In[40]:


def get_recommendations(title):
    indice = indices[title]
    sim_scores = list(enumerate(user_profile[indice]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[41]:


get_recommendations('Forrest Gump').head(10)


# In[42]:


get_recommendations('The Confessional').head(10)


# ### Content based RS: Using movie description, keywords, genres, taglines, director and cast

# On merging the movie metadata set, keywords and credits dataset

# In[43]:


metadata_movie['id'] = metadata_movie['id'].astype('int')
keywords_dataset['id'] = keywords_dataset['id'].astype('int')
credits_dataset['id'] = credits_dataset['id'].astype('int')


# In[44]:


movies = metadata_movie.merge(credits_dataset, on='id')
movies = movies.merge(keywords_dataset, on= 'id')


# In[45]:


subset_movies = movies[movies['id'].isin(tags_dataset)]


# In[46]:


subset_movies.shape


# Now we have all our cast, genres and keywords in one one dataframe. We have to clean the data.\
# 1- Crew - We will choose only the directors from the crew as others do not contribute to the feel of the movie.\
# 2- Cast - We do not have to go through all the actors as lesser known actors do not influence the opinion of the people towards watching a movie, hence we will be choosing top 5 actors that appear in the cast list.

# In[47]:


subset_movies['cast'] = subset_movies['cast'].apply(lambda x: literal_eval(str(x)))
subset_movies['crew'] = subset_movies['crew'].apply(lambda x: literal_eval(str(x)))
subset_movies['keywords'] = subset_movies['keywords'].apply(lambda x: literal_eval(str(x)))
subset_movies['cast_size'] = subset_movies['cast'].apply(lambda x: len(x))
subset_movies['crew_size'] = subset_movies['crew'].apply(lambda x: len(x))


# In[48]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[49]:


subset_movies['director'] = subset_movies['crew'].apply(get_director)
subset_movies['cast'] = subset_movies['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
subset_movies['cast'] = subset_movies['cast'].apply(lambda x: x[:5] if len(x) >= 5 else x)


# In[50]:


subset_movies['keywords'] = subset_movies['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[51]:


subset_movies['cast'] = subset_movies['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
subset_movies['director'] = subset_movies['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
subset_movies['director'] = subset_movies['director'].apply(lambda x: [x,x, x])


# In[52]:


#calculate the frequency count of every keyword that appears in the dataset

subset = subset_movies.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
subset.name = 'keyword'
subset = subset.value_counts()
subset[:5]


# In[53]:


subset = subset[subset > 1]


# Convert every word to its stem so that words such as Dogs and Dog are considered the same.

# In[54]:


from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()


# In[55]:


# def filter_keywords(x):
#     words = []
#     for i in x:
#         for i in subset:
#             words.append(i)
#     return words


# In[57]:


# subset_movies['keywords'] = subset_movies['keywords'].apply(filter_keywords)
# subset_movies['keywords'] = subset_movies['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
#subset_movies['keywords'] = subset_movies['keywords'].apply(lambda x: [str.lower(str(i).replace(" ", "")) for i in x])


# In[58]:


#subset_movies['collective'] = subset_movies['keywords'] + subset_movies['cast'] + subset_movies['director'] + subset_movies['genres']
#subset_movies['collective'] = subset_movies['collective'].apply(lambda x: ' '.join(x))


# In[61]:


# def improved_recommendations(title):
#     idx = indices[title]
#     sim_scores = list(enumerate(user_profile[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:26]
#     movie_indices = [i[0] for i in sim_scores]
    
#     movies = subset_movies.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
#     vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
#     vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
#     C = vote_averages.mean()
#     m = vote_counts.quantile(0.60)
#     qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & 
#                        (movies['vote_average'].notnull())]
#     qualified['vote_count'] = qualified['vote_count'].astype('int')
#     qualified['vote_average'] = qualified['vote_average'].astype('int')
#     qualified['wr'] = qualified.apply(weighted_rating, axis=1)
#     qualified = qualified.sort_values('wr', ascending=False).head(10)
#     return qualified


# In[63]:


# improved_recommendations('The Dark Knight')


# Using Surprise library that uses extremely powerful algorithms like Singular Value Decomposition (SVD) to minimize RMSE (Root Mean Square Error) and give great recommendations

# In[65]:


pip install surprise


# In[67]:


# surprise reader API to read the dataset
from surprise import Reader, Dataset, SVD
reader = Reader()


# In[68]:


data = Dataset.load_from_df(ratings_dataset[['userId', 'movieId', 'rating']], reader)


# In[107]:


#data.split(n_folds=5)


# In[77]:


from surprise.model_selection import cross_validate
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)


# In[78]:


trainset = data.build_full_trainset()


# In[80]:


svd.fit(trainset)


# In[81]:


ratings_dataset[ratings_dataset['userId'] == 1]


# In[108]:


svd.predict(2, 302)


# ## Hybrid recommendation system

# In[83]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[95]:


id_map = pd.read_csv('C:/Users/sameer.sakkhari/Desktop/Manasa/CS698/dataset_project/tags_dataset.csv')[['movieId','tmdbId']]


# In[96]:


id_map


# In[97]:


id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)


# In[98]:


id_map.columns = ['movieId', 'id']
id_map = id_map.merge(subset_movies[['title', 'id']], on='id').set_index('title')


# In[99]:


indices_map = id_map.set_index('id')


# In[103]:


def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    sim_scores = list(enumerate(user_profile[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = subset_movies.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'release_date', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)


# In[104]:


hybrid(1, 'Avatar')


# In[105]:


hybrid(5000, 'Avatar')


# In[106]:


hybrid(3423, "The Terminator")


# In[ ]:




