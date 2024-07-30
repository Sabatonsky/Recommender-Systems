# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# userId, movieId, rating, timestamp
# int, int, float (1.0 - 5.0, with halfs), date

#!wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
#!unzip ml-1m.zip

# Define file directories
MOVIELENS_DIR = 'ml-1m'
RATING_DATA_FILE = 'ratings.dat'

# Define csv files to be saved into
RATINGS_CSV_FILE = 'ratings.csv'

ini_path = os.path.join(MOVIELENS_DIR, RATING_DATA_FILE)
train_path = 'train_ratings.csv'
test_path = 'test_ratings.csv'

def preprocessing(ini_path, train_path, test_path, max_user, max_movie, load=True, save=True):
  if load == True and os.path.exists(train_path):
    print('Loading dataframe from file...')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
  else:
    print('File was not found, preprocessing...')
    df = pd.read_csv(ini_path,
                      sep='::',
                      engine='python',
                      encoding='latin-1',
                      names=['userId', 'movieId', 'rating', 'timestamp'])
    df.drop(columns=['timestamp'], inplace = True)
    df.userId = df.userId - 1
    unique_movie_ids = set(df.movieId.values)
    num_movies = len(unique_movie_ids)
    movie2idx = dict(zip(unique_movie_ids, range(num_movies)))
    df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)
    s_userId = df.groupby(['userId'])['movieId'].count().nlargest(max_user, 'first')
    s_movieId = df.groupby(['movieId'])['userId'].count().nlargest(max_movie, 'first')
    df = df.loc[df.movieId.isin(s_movieId.index)]
    df = df.loc[df.userId.isin(s_userId.index)]
    unique_user_ids = set(df.userId.values)
    unique_movie_ids = set(df.movieId.values)
    user2idx = dict(zip(unique_user_ids, range(max_user)))
    movie2idx = dict(zip(unique_movie_ids, range(max_movie)))
    df['user_idx'] = df.apply(lambda row: user2idx[row.userId], axis=1)
    df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)
    train, test = train_test_split(df, test_size=0.2)
    if save == True:
      print('Saving dataframe...')
      train.to_csv(train_path)
      test.to_csv(test_path)
  return train, test

train, test = preprocessing(ini_path, train_path, test_path, 6000, 2000)

def update_data(row):
  i = int(row.user_idx)
  j = int(row.movie_idx)
  if i not in user2movie:
    user2movie[i] = [j]
  else:
    user2movie[i].append(j)

  if j not in movie2user:
    movie2user[j] = [i]
  else:
    movie2user[j].append(i)

  usermovie2rating[(i, j)] = row.rating

def update_data_test(row):
  i = int(row.user_idx)
  j = int(row.movie_idx)
  usermovie2rating_test[(i, j)] = row.rating

if not os.path.exists('user2movie.json') or \
  not os.path.exists('movie2user.json') or \
  not os.path.exists('usermovie2rating.json') or \
  not os.path.exists('usermovie2rating_test.json'):
  user2movie = {}
  movie2user = {}
  usermovie2rating = {}
  usermovie2rating_test = {}
  train.apply(update_data, axis=1)
  test.apply(update_data_test, axis=1)
  with open('user2movie.json', 'wb') as f:
    pickle.dump(user2movie, f)
  with open('movie2user.json', 'wb') as f:
    pickle.dump(movie2user, f)
  with open('usermovie2rating.json', 'wb') as f:
    pickle.dump(usermovie2rating, f)
  with open('usermovie2rating_test.json', 'wb') as f:
    pickle.dump(usermovie2rating_test, f)
else:
  with open('user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)
  with open('movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)
  with open('usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)
  with open('usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

n = np.max(list(user2movie.keys())) + 1
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m) in usermovie2rating_test.items()])
m = max(m1, m2) + 1

k = 25
limit = 5
neighbors = [ [] for _ in range(n) ] #List of knn
averages = [0]*n #List of means
deviations = [ {} for _ in range(n) ] #List of ratings - mean
sigmas = [0]*n
users = [ () for _ in range(n) ]

for i in range(m):
  users_i = movie2user[i] #All films that were rated by user
  users_i_set = set(users_i) #for intersection check
  ratings_i = {user:usermovie2rating[(user, i)] for user in users_i}
  #all ratings of user in format {movie:rating}
  avg_i = np.mean(list(ratings_i.values())) #mean them
  dev_i = {user:(rating - avg_i) for user, rating in ratings_i.items()}
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))
  averages[i] = avg_i
  deviations[i] = dev_i
  sigmas[i] = sigma_i
  users[i] = users_i_set

for i in range(m):
  if i % 100 == 0:
    print('i:', i)
  for j in range(m):
    if i < j:
      common_users = (users[i] & users[j])
      if len(common_users) > limit:
        num = sum(deviations[i][n]*deviations[j][n] for n in common_users)
        if num != 0:
          w_ij = num / (sigmas[i] * sigmas[j])
        else:
          w_ij = 0
        neighbors[i].append((w_ij, j))
        neighbors[j].append((w_ij, i))
  if len(neighbors[i]) > k:
    neighbors[i] = sorted(neighbors[i], reverse=True)[:k]

X_test = np.array([test.movie_idx, test.user_idx]).T
Y_test = np.array([test.rating]).T
X_train = np.array([train.movie_idx, train.user_idx]).T
Y_train = np.array([train.rating]).T

def predict(X, Y, k):
  P = np.zeros(X.shape[0])
  for i in range(X.shape[0]):
    cur_user = X[i, 1]
    cur_movie = X[i, 0]
    num = 0
    den = 0
    for w, j in neighbors[cur_movie]:
      try:
        num += w*deviations[j][cur_user]
        den += abs(w)
      except KeyError:
        pass
    if den == 0:
       averages[cur_movie]
    else:
      P[i] = num / den + averages[cur_movie]
      P[i] = min(5, P[i])
      P[i] = max(0.5, P[i])
  return P

k = 25
S_test = predict(X_test, Y_test, k)
S_train = predict(X_train, Y_train, k)

MSE_test = mean_squared_error(Y_test, S_test)
MSE_train = mean_squared_error(Y_train, S_train)
print(MSE_test)
print(MSE_train)

