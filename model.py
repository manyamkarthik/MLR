import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the California Housing dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target  # Add the target variable

# Prepare the data
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with Polynomial Features, StandardScaler, and Ridge Regression
# Pipeline is used to apply a list of operations sequentially

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),  # Experiment with different degrees
    ('scaler', StandardScaler()),            # Scale the features
    ('ridge', Ridge(alpha=1.0))               # Experiment with different alpha values
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model
filename = 'california_housing_model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))  # Save the whole pipeline

print(f"Model saved to {filename}")