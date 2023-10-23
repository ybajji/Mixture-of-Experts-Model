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




class MLPGateModel:
    def __init__(self, num_experts, input_dim):
        self.gate_model = MLPClassifier(hidden_layer_sizes=(100, num_experts), activation='logistic', max_iter=1000)

    def train(self, X, y):
        self.gate_model.fit(X, y)

    def get_expert_weights(self, X):
        gate_probabilities = self.gate_model.predict_proba(X)
        return gate_probabilities

class MixtureExperts:
    def __init__(self, num_experts, num_classes):
        self.experts = [ExpertDecisionTree(num_classes) for _ in range(num_experts)]
        self.gating = MLPGateModel(num_experts=num_experts, input_dim=X_train.shape[1])
        self.num_experts = num_experts
        self.num_classes = num_classes

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.num_experts, random_state=42).fit(X)
        labels = kmeans.labels_
        
        # Train the gating network with X and labels as targets
        self.gating.train(X, labels)

        data_per_expert = [None] * self.num_experts  # Initialize an empty list

        # Divide data among experts based on cluster labels
        for i in range(self.num_experts):
            idx = np.where(labels == i)  # Get indices of data points in the current cluster
            X_expert, y_expert = X[idx], y[idx]
            data_per_expert[i] = (X_expert, y_expert)

        # Train experts with gating weights
        for idx, expert in enumerate(self.experts):
            X_expert, y_expert = data_per_expert[idx]
            expert.fit(X_expert, y_expert)
            
            
    def predict(self, X):
       gating_weights = self.gating.get_expert_weights(X)
       expert_outputs = []

    # Wählen Sie die Experten basierend auf den Wahrscheinlichkeiten aus
       selected_experts = np.argmax(gating_weights, axis=1)
       print(selected_experts)

       expert_predictions = []

       for i in range(X.shape[0]):
           gating_weight = gating_weights[i]
           expert_idx = np.argmax(gating_weight)
           expert = self.experts[expert_idx]
  
           expert_outputs_expert = expert.predict(X[i].reshape(1, -1))
           expert_predictions.append(expert_outputs_expert)

       expert_predictions = np.vstack(expert_predictions)  # Stapeln Sie die Experten-Vorhersagen vertikal
       final_predictions = np.argmax(expert_predictions, axis=1)  # Wählen Sie die Klasse mit der höchsten Wahrscheinlichkeit

       return final_predictions
         


    
                             
   

      

               






# Instantiate and train the model
moe = MixtureExperts(num_experts=3, num_classes=3)
moe.fit(X_train, y_train)

# Evaluate the model
y_pred = moe.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
f1= f1_score(y_test, y_pred,average='macro')
print("+++++++++++++Accuracy++++++++++++:", accuracy)
print("+++++++++++++macro F1 score++++++++++++:",f1) 

