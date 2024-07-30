# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset

#Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    global_mean = train['rating'].mean()
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
    global_mean = train['rating'].mean()
    if save == True:
      print('Saving dataframe...')
      train.to_csv(train_path)
      test.to_csv(test_path)
  return train, test, global_mean

train, test, global_mean = preprocessing(ini_path, train_path, test_path, 6000, 2000)
train['rating'] -= global_mean
test['rating'] -= global_mean

n = 6000
m = 2000

class MatrixFactorization(torch.nn.Module):
  def __init__(self, n, m, k):
    super().__init__()
    self.w = torch.nn.Embedding(n, k)
    self.u = torch.nn.Embedding(m, k)
    self.b = torch.nn.Embedding(n, 1)
    self.c = torch.nn.Embedding(m, 1)

  def forward(self, user, movie):
    pred = self.b(user) + self.c(movie)
    pred += (self.w(user) * self.u(movie)).sum(dim=1, keepdim=True)
    return pred.squeeze()

class RMSELoss(nn.Module):
  def __init__(self, eps=1e-8):
    super().__init__()
    self.mse = nn.MSELoss()
    self.eps = eps

  def forward(self, yhat, y):
    loss = torch.sqrt(self.mse(yhat, y) + self.eps)
    return loss

class CustomDataset(Dataset):
    def __init__(self, users, movies, labels):
        self.users = users
        self.movies = movies
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        user = self.users[index]
        movie = self.movies[index]
        label = self.labels[index]
        return user, movie, label

k = 10
batch_size = 1000
num_epochs = 25
model = MatrixFactorization(n, m, k)
criterion = RMSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9, weight_decay=10e-6)

users_test = torch.as_tensor(test.user_idx, dtype=torch.int64)
movies_test = torch.as_tensor(test.movie_idx, dtype=torch.int64)
labels_test = torch.as_tensor(test.rating, dtype=torch.float32)
users_train = torch.as_tensor(train.user_idx, dtype=torch.int64)
movies_train = torch.as_tensor(train.movie_idx, dtype=torch.int64)
labels_train = torch.as_tensor(train.rating, dtype=torch.float32)

test_dataset = CustomDataset(users_test, movies_test, labels_test)
train_dataset = CustomDataset(users_train, movies_train, labels_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
n_total_steps = len(train_loader)
epoch_loss = []

for epoch in range(num_epochs):
  losses = []
  for i, (users, movies, labels) in enumerate(train_loader):
    #flatten
    users = users.to(device)
    movies = movies.to(device)
    labels = labels.to(device)

    #forward
    outputs = model(users, movies)
    loss = criterion(outputs, labels)
    losses.append(loss.item())

    #backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  epoch_loss.append(np.mean(losses))
  print(f'epoch {epoch+1} / {num_epochs}, train loss = {epoch_loss[-1]:.4f}')

plt.plot(epoch_loss)
plt.savefig('losses')
plt.clf()

#test
with torch.no_grad():
  losses = []
  for i, (users, movies, labels) in enumerate(test_loader):
    #flatten
    users = users.to(device)
    movies = movies.to(device)
    labels = labels.to(device)

    #forward
    outputs = model(users, movies)
    loss = criterion(outputs, labels)
    losses.append(loss.item())

  print(f'test loss = {np.mean(losses):.4f}')
