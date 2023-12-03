import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from torch import nn, optim

class Expert:
    def __init__(self):
        self.model = DecisionTreeRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class GatingNetwork(nn.Module):
    def __init__(self, n_inputs, n_experts):
        super(GatingNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, n_experts, dtype=torch.float64),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, n_inputs, n_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = [Expert() for _ in range(n_experts)]
        self.gating_network = GatingNetwork(n_inputs, n_experts)
        self.n_expert = n_experts

        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, n_experts, dtype=torch.float64)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float64)
        expert_preds = torch.tensor([expert.predict(x) for expert in self.experts], dtype=torch.float64).permute(1, 0)
        gating_probs = self.gating_network(x)
        preds = expert_preds * gating_probs
        final_pred = torch.sum(preds, dim=1)
        return final_pred

    def train(self, x, y):
        y_train = torch.tensor(y, dtype=torch.float64)
        X_train = torch.tensor(x, dtype=torch.float64)

        kmeans = KMeans(n_clusters=self.n_expert, n_init=10)
        clusters = kmeans.fit_predict(X_train)

        for i in range(self.n_expert):
            self.experts[i].fit(X_train[clusters == i], y_train[clusters == i])

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 100, loss.item()))

 # Use the forward method for regression predictions
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Specify the test size and random state
test_size = 0.2
random_state = 42

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

  




# Create and train the regression model
model = MixtureOfExperts(n_inputs=X.shape[1], n_experts=3)
model.train(X_train, y_train)

# Make predictions using the model and convert to NumPy
y_pred = model.forward(X_test).detach().numpy()

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

