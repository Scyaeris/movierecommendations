
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# load csv files
movies_csv = pd.read_csv('movies/movies.csv')
ratings_csv = pd.read_csv('movies/ratings.csv')

# create list of unique movies
unique_movies = movies_csv['movieId'].unique()

# create a user-movie matrix
user_movie_matrix = np.zeros((ratings_csv['userId'].max()+1, len(unique_movies)))

# fill matrix with rating values
for row in range(58098):
    user_movie_matrix[row[1]-1, row[2]-1] = row[3]

# create train and test sets
train_set = user_movie_matrix[:int(0.8*len(user_movie_matrix))]
test_set = user_movie_matrix[int(0.8*len(user_movie_matrix)):]

# define the model
class RecommenderNet(nn.Module):
    def __init__(self):
        super(RecommenderNet, self).__init__()
        self.fc1 = nn.Linear(len(unique_movies), 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, len(unique_movies))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# create model and optimizer
model = RecommenderNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
for epoch in range(200):
    # convert numpy arrays to torch tensors
    inputs = torch.from_numpy(train_set).float()
    targets = torch.from_numpy(train_set).float()

    # forward pass
    outputs = model(inputs)
    loss = torch.nn.MSELoss()(outputs, targets)

    # backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 200, loss.item()))

# evaluate the model
with torch.no_grad():
    inputs = torch.from_numpy(test_set).float()
    targets = torch.from_numpy(test_set).float()
    outputs = model(inputs)
    loss = torch.nn.MSELoss()(outputs, targets)
    print('Test Loss: {:.4f}'.format(loss.item()))

# make predictions
predictions = outputs.data.numpy()

# create list of top 10 movie recommendations for each user
top_10_recommendations = []

for user in range(predictions.shape[0]):
    user_recommendations = predictions[user,:]
    top_10_indices = user_recommendations.argsort()[-10:][::-1]
    top_10_movies = unique_movies[top_10_indices]
    top_10_recommendations.append(top_10_movies)

print(top_10_recommendations)