import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Read original dataset
dataset = pd.read_csv("data/petrol_consumption.csv")

# Extract features and target
x = dataset.drop('Petrol_Consumption', axis = 1) # Features
y = dataset['Petrol_Consumption']  # Target


# Splitting the dataset into training and testing set (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 28)

# Create a Random Forest Regressor model
rfr = RandomForestRegressor(n_estimators=10, random_state=42)

# Train the model
rfr.fit(x_train, y_train)

# Predict on the test set
y_pred = rfr.predict(x_test)

# Calculate the mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model to disk
joblib.dump(rfr, "rf_model.sav")
