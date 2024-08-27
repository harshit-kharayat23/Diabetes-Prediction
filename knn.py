import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("D:/Diabetes prediction/diabetes.csv")

# Split into X and Y
X = data.drop("Outcome", axis=1)
Y = data["Outcome"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['euclidean', 'manhattan', 'chebyshev']  # Distance metric
}

# Initialize k-NN classifier
knn = KNeighborsClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform GridSearchCV to find the best parameters
grid_search.fit(X_train_scaled, Y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)

# Train the model with the best parameters
best_knn_model = grid_search.best_estimator_

# Predict on the test data
knn_predictions = best_knn_model.predict(X_test_scaled)

# Calculate accuracy
knn_accuracy = accuracy_score(Y_test, knn_predictions)
print(f'k-NN Accuracy with GridSearchCV: {knn_accuracy:.2f}')

# Print classification report
print('Classification Report:')
print(classification_report(Y_test, knn_predictions))
