from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from torch import nn, optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    def __init__(self, n_inputs, n_experts, num_classes):
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
        num_chunks = preds.size(1) // self.n_expert
        final_pred_list = []
        for i in range(num_chunks):
            start_idx = i * self.num_classes
            end_idx = (i + 1) * self.num_classes
            chunk_sum = torch.sum(preds[:, start_idx:end_idx], dim=1)
            final_pred_list.append(chunk_sum)

        final_pred = torch.stack(final_pred_list, dim=1)
        return final_pred



    def predict(self, x):
        return self.forward(x).detach().numpy().argmax(axis=1)

    def train(self, x, y):
        y_train = torch.tensor(y, dtype=torch.long)
        X_train = torch.tensor(x, dtype=torch.float64)

        kmeans = KMeans(n_clusters=self.n_expert,n_init=10)
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


#---------------------------------------------------------------------------------------------------#
digits =datasets.load_iris()
X = digits.data
y = digits.target

# Aufteilen der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modellinstanz erstellen
model_digits = MixtureOfExperts(n_inputs=X.shape[1], n_experts=6, num_classes=3)

# Modell trainieren
model_digits.train(X_train, y_train)

# Vorhersagen für das Testset machen
y_pred= model_digits.predict(X_test)
accuracy_digits = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy_digits)
