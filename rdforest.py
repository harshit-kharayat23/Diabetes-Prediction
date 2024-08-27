# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv("D:/Diabetes prediction/diabetes.csv")

# Split into X and Y
X = data.drop("Outcome", axis=1)
Y = data["Outcome"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Perform Grid Search Cross-Validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, Y_train)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Predict on the test data using the best model
rf_predictions = best_rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(Y_test, rf_predictions)
print(f'Random Forest Accuracy on Test Set: {rf_accuracy:.2f}')

# Print classification report
print('Classification Report:')
print(classification_report(Y_test, rf_predictions))
