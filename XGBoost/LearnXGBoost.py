import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, cross_val_score

DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"

def main():
	global DATASET_URL

	# This dataset has no column names hence setting header as 'None'
	data = pd.read_csv(DATASET_URL, header=None)
	print("Dataset Shape: ", data.shape)

	# Print top 5 rows of dataset
	print(data.head())

	# Split the dataset as predictors and target
	data = data.values
	X, y = data[:, :-1], data[:, -1]

	# Define an XGBoost Regressor
	model = xgboost.XGBRegressor()

	# Define a repeated k fold iterator
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

	# Get the score of folding
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

	# Score will negative so we need to take absolute value
	scores = np.absolute(scores)

	print(f"Mean MAE: {scores.mean():.3f} | STD: {scores.std():.3f}")

if __name__ == "__main__":
	main()