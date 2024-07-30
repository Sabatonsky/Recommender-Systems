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

x_train = torch.sparse_coo_tensor([list(train.user_idx), list(train.movie_idx)], list(train.rating), (n, m))
x_test = torch.sparse_coo_tensor([list(test.user_idx), list(test.movie_idx)], list(test.rating), (n, m))

class Autoencoder(torch.nn.Module):
  def __init__(self, m, h):
    super().__init__()
    self.l1 = nn.Linear(m, h)
    self.l2 = nn.Linear(h, m)
    self.drop1 = nn.Dropout(p=0.7)
    self.drop2 = nn.Dropout(p=0.5)

  def forward(self, x):
    x = self.drop1(x)
    x = self.l1(x)
    x = torch.tanh(x)
    x = self.drop2(x)
    out = self.l2(x)
    return out

class RMSELoss(nn.Module):
  def __init__(self, eps=1e-8):
    super().__init__()
    self.mse = nn.MSELoss()
    self.eps = eps

  def forward(self, yhat, y):
    mask = torch.where(y == 0, 0, 1)
    masked_yhat = mask * yhat
    masked_n = mask.sum()
    loss = torch.sqrt(((masked_yhat - y)**2).sum() / masked_n + self.eps)
    return loss

class CustomDataset(Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index].to_dense()
        return x

batch_size = 128
h = 700
num_epochs = 20
model = Autoencoder(m, h)
criterion = RMSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

test_dataset = CustomDataset(x_test)
train_dataset = CustomDataset(x_train)

s_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
n_total_steps = len(train_loader)
epoch_loss = []
test_epoch_loss = []

for epoch in range(num_epochs):
  losses = []
  test_losses = []
  for i, (s_x_train, x_train, x_test) in enumerate(zip(s_train_loader, train_loader, test_loader)):
    #flatten
    s_x_train = s_x_train.to(device)
    x_train = x_train.to(device)
    x_test = x_test.to(device)

    #forward
    outputs = model(s_x_train)
    loss = criterion(outputs, s_x_train)
    losses.append(loss.item())

    #backwards
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      outputs = model(x_train)
      test_loss = criterion(outputs, x_test)
      test_losses.append(test_loss.item())

  epoch_loss.append(np.mean(losses))
  test_epoch_loss.append(np.mean(test_losses))

  print(f'epoch {epoch+1} / {num_epochs}, train loss = {epoch_loss[-1]:.4f}, test loss = {test_epoch_loss[-1]:.4f}')

plt.plot(epoch_loss, label='train_loss')
plt.plot(test_epoch_loss, label='test_loss')
plt.legend()
plt.savefig('train loss')
plt.clf()
