import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

#import the ratings and movies data set
ratings = pd.read_csv('ratings.csv', encoding = 'utf-8')
movies = pd.read_csv('movies.csv', encoding = 'utf-8')

#join ratings and movies data set
data = pd.merge(movies, ratings, on='movieId')

#visualizing the ratings
sns.set_style('ticks')
sns.catplot(x='rating', data=data, kind='count', aspect=2.5)
plt.show()

#Split data into training, validation and test set
from sklearn.model_selection import train_test_split
training_data, test_data = train_test_split(data, test_size=0.2)
training_data, validation_data = train_test_split(training_data, test_size=0.2)

#create the dataloaders
from torch.utils.data import DataLoader, TensorDataset

#create Tensor datasets from Partioned data

train_tensor_ds = TensorDataset(torch.FloatTensor(training_data.values), torch.LongTensor(training_data.values))
valid_tensor_ds = TensorDataset(torch.FloatTensor(validation_data.values), torch.LongTensor(validation_data.values))
test_tensor_ds = TensorDataset(torch.FloatTensor(test_data.values), torch.LongTensor(test_data.values))

#create data loaders from Tensor datasets

train_loader = DataLoader(dataset=train_tensor_ds, batch_size=32, shuffle=True)
validation_loader = DataLoader(dataset=valid_tensor_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_tensor_ds, batch_size=32, shuffle=True)

#Define a Convolutional Neural Network

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        #Define 2D Convolution Layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        #Fully Connected Layer
        self.fc_layer = nn.Sequential(
            nn.Linear(20*20*32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
            )

    #Define from_input_to_hidden
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 20*20*32)
        x = self.fc_layer(x)
        return x


# Instantiate the model
model = ConvNet()
# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Train the Model

n_epochs = 10

for epoch in range(1, n_epochs+1):
    train_losses = []
    valid_losses = []
    # Train the model
    for batch, (inputs, labels) in enumerate(train_loader, 1):
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model.forward(inputs)
        # Calculate Loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Store the loss
        train_losses.append(loss.item())
        # Validate Model
    with torch.no_grad():
        valid_loss = 0.0
        for inputs, labels in validation_loader:
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
        valid_loss /= len(validation_loader)
        valid_losses.append(valid_loss)
    # Print training losses
    print(f'Epoch: {epoch} \tTraining Loss: {np.mean(train_losses):.5f} \tValidation Loss: {np.mean(valid_losses):.5f}')

#Test the Model

test_losses = []

#Test the model
with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in test_loader:
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
    test_loss /= len(test_loader)
    test_losses.append(test_loss)

print(f'Test Loss: {np.mean(test_losses):.5f}')

# Make Predictions

# Make Predictions on the test set
with torch.no_grad():
    preds = []

# Get the predicted values
    for batch, (inputs, labels) in enumerate(test_loader, 1):
        pred = model.forward(inputs)
        preds.append(pred)

# Flatten the list of predictions
preds = [pred.data.numpy()[0] for pred in preds]
true_labels = [label.data.numpy()[0] for label in test_loader.dataset.tensors[1]]