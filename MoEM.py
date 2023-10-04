import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class ExpertDecisionTree:
    def __init__(self, num_classes):
        self.model = DecisionTreeClassifier()
        self.num_classes = num_classes

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        proba = self.model.predict_proba(X)
        # Ensure the output has shape (n_samples, num_classes)
        full_proba = np.zeros((proba.shape[0], self.num_classes))
        class_indices = self.model.classes_.astype(int)
        full_proba[:, class_indices] = proba
        print(full_proba)
        return full_proba


# Define the gating network
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

class MixtureExperts:
    def __init__(self, num_experts, num_classes):
        self.experts = [ExpertDecisionTree(num_classes) for _ in range(num_experts)]
 
        self.num_experts = num_experts
        self.num_classes = num_classes

    def fit(self, X, y):
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.num_experts, random_state=42).fit(X)
        labels = kmeans.labels_
        
        # Divide data among experts based on cluster labels
        data_per_expert = [None] * self.num_experts  # Initialize an empty list
        for i in range(self.num_experts):
            idx = np.where(labels == i)  # Get indices of data points in current cluster
            X_expert, y_expert = X[idx], y[idx]
            data_per_expert[i] = (X_expert, y_expert)
        
        # Train experts
        for idx, expert in enumerate(self.experts):
            X_expert, y_expert = data_per_expert[idx]
            expert.fit(X_expert, y_expert)

    def predict(self, X):
        # Obtain predictions from each expert
        expert_outputs = np.stack([expert.predict(X) for expert in self.experts], axis=0)
        # Simple averaging of expert predictions
        final_predictions = np.mean(expert_outputs, axis=0)
        return np.argmax(final_predictions, axis=1)



# Instantiate and train the model
moe = MixtureExperts(num_experts=3, num_classes=3)
moe.fit(X_train, y_train)

# Evaluate the model
y_pred = moe.predict(X_test)  # Pass verbose=True to print the predictions and weights

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average='macro')
print("Macro F1-score:", f1)