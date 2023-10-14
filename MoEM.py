import torch.nn as nn
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

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

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        proba = self.model.predict_proba(X)
        # Ensure the output has shape (n_samples, num_classes)
        full_proba = np.zeros((proba.shape[0], self.num_classes))
        class_indices = self.model.classes_.astype(int)
        full_proba[:, class_indices] = proba
        return full_proba




class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = MLPClassifier(hidden_layer_sizes=(100, num_experts), activation='relu', max_iter=1000)

    def train_gating(self, X, labels):
        self.fc.fit(X, labels)

    def predict(self, X):
        return self.fc.predict_proba(X)
class MixtureExperts:
    def __init__(self, num_experts, num_classes):
        self.experts = [ExpertDecisionTree(num_classes) for _ in range(num_experts)]
        self.gating = GatingNetwork(input_size=X_train.shape[1], num_experts=num_experts)
        self.num_experts = num_experts
        self.num_classes = num_classes

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.num_experts, random_state=42).fit(X)
        labels = kmeans.labels_
        data_per_expert = [None] * self.num_experts
        

        # Train the gating network with X and labels as targets
        self.gating.train_gating(X, labels)
        # Divide data among experts based on cluster labels
        data_per_expert = [None] * self.num_experts  # Initialize an empty list
        for i in range(self.num_experts):
            idx = np.where(labels == i)  # Get indices of data points in the current cluster
            X_expert, y_expert = X[idx], y[idx]
            data_per_expert[i] = (X_expert, y_expert)
        # Train experts with gating weights
        for idx, expert in enumerate(self.experts):
            X_expert, y_expert = data_per_expert[idx]       
            expert.fit(X_expert, y_expert)
    def predict(self, X):
        gating_weights = self.gating.predict(X)
        print("+++++++++++++++++ gatting weight++++++++++++++++++++++++")
        print(gating_weights)
        expert_outputs = []

        for idx, expert in enumerate(self.experts):
            expert_outputs_expert = expert.predict(X)
            weighted_expert_outputs = expert_outputs_expert * gating_weights[:, idx][:, np.newaxis]
            expert_outputs.append(weighted_expert_outputs)
            print("++++++++++++++++++++ expert outputs +++++++++++")
            print(expert_outputs_expert)
        final_predictions = np.sum(expert_outputs, axis=0)
        print("+++++++++++++++++++pred ++++++++++++++++++++++")
        print(final_predictions)
        return np.argmax(final_predictions, axis=1)






# Instantiate and train the model
moe = MixtureExperts(num_experts=2, num_classes=3)
moe.fit(X_train, y_train)

# Evaluate the model
y_pred = moe.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1= f1_score(y_test, y_pred,average='macro')
print("+++++++++++++Accuracy++++++++++++:", accuracy)
print("+++++++++++++macro F1 score++++++++++++:",f1) 