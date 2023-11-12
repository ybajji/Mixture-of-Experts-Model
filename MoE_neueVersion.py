import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from torch import nn, optim

class Expert:
    def __init__(self, num_classes):
        self.model = DecisionTreeClassifier()
        self.num_classes = num_classes

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        proba = self.model.predict_proba(X)
        full_proba = np.zeros((proba.shape[0], self.num_classes), dtype=np.float64)
        class_indices = self.model.classes_.astype(int)
        full_proba[:, class_indices] = proba
        print(full_proba)
        return full_proba

class GatingNetwork(nn.Module):
    def __init__(self, n_inputs, n_experts, num_classes=3):
        super(GatingNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, n_experts * num_classes, dtype=torch.float64),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, n_inputs, n_experts, num_classes):
        super(MixtureOfExperts, self).__init__()
        self.experts = [Expert(num_classes) for _ in range(n_experts)]
        self.gating_network = GatingNetwork(n_inputs, n_experts, num_classes)
        self.n_expert = n_experts
        self.num_classes = num_classes

        # Set the data type for the linear layers
        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, 30, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(30, n_experts * num_classes, dtype=torch.float64),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float64)
        
        expert_preds = torch.tensor([expert.predict(x) for expert in self.experts], dtype=torch.float64).permute(1, 0, 2).reshape(-1, self.num_classes * self.n_expert)
        print(expert_preds)
        gating_probs = self.gating_network(x)
        print(gating_probs)
        preds = expert_preds * gating_probs
        print("------------------------------------------------")
        print(preds)
        print("------------------------------------------------")
        final_pred = torch.stack([
            torch.sum(preds[:, 0::3], dim=1),
            torch.sum(preds[:, 1::3], dim=1),
            torch.sum(preds[:, 2::3], dim=1),
        ], dim=1)
        return final_pred




    def predict(self, x):
        return self.forward(x).detach().numpy().argmax(axis=1)

    def train(self, x, y):
        y_train = torch.tensor(y, dtype=torch.long)
        X_train = torch.tensor(x, dtype=torch.float64)

        kmeans = KMeans(n_clusters=3,n_init=10)
        clusters = kmeans.fit_predict(X_train)

        for i in range(self.n_expert):
            self.experts[i].fit(X_train[clusters == i], y_train[clusters == i])

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 100, loss.item()))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MixtureOfExperts(n_inputs=4, n_experts=3, num_classes=3)
model.train(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
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



class TrainableGateModel:
    def __init__(self, input_dim, num_experts):
        self.gate_model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000)
        self.input_dim = input_dim
        self.num_experts = num_experts

    def train(self, X, labels):
        self.gate_model.fit(X, labels)

    def get_expert_weights(self, X):
        gate_probabilities = self.gate_model.predict_proba(X)
        return gate_probabilities

class MixtureExperts:
    def __init__(self, num_experts, num_classes):
        self.experts = [ExpertDecisionTree(num_classes) for _ in range(num_experts)]
        self.num_experts = num_experts
        self.num_classes = num_classes

    def fit(self, X, y, gating_weights):
       

        for i in range(self.num_experts):
            expert = self.experts[i]
            idx = np.where(gating_weights == i)
            X_expert, y_expert = X[idx], y[idx]
            expert.fit(X_expert, y_expert)

    def predict(self, X, gating_weights):
        expert_outputs = []

        for i in range(X.shape[0]):
            expert_idx = np.argmax(gating_weights[i])
            expert = self.experts[expert_idx]

            expert_outputs_expert = expert.predict(X[i].reshape(1, -1))
            expert_outputs.append(expert_outputs_expert)

        final_predictions = np.argmax(expert_outputs, axis=1)

        return final_predictions


# Beispielaufruf:
num_experts = 3
input_dim = X_train.shape[1]

# Erstellen Sie Ihre trainierbare Gating-Unit und trainieren Sie sie mit X_train und Cluster-Labels
gating_unit = TrainableGateModel(input_dim, num_experts)
kmeans = KMeans(n_clusters=num_experts, random_state=42).fit(X_train)
labels = kmeans.labels_
gating_unit.train(X_train, labels)

# Erstellen Sie Ihr MoE-Modell
moe = MixtureExperts(num_experts=num_experts, num_classes=3)

# Trainieren Sie Ihr MoE-Modell mit den Gating-Gewichten
moe.fit(X_train, y_train, labels)
y_pred = moe.predict(X_test, gating_unit.get_expert_weights(X_test))

print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print("+++++++++++++Accuracy++++++++++++:", accuracy)
print("+++++++++++++macro F1 score++++++++++++:", f1)

# update 30 oct
