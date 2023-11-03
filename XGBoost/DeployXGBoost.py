import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xbg
import pickle

DATASET_URL = "../Dataset/pima-indians-diabetes.data.csv"

def main():
	data = np.loadtxt(DATASET_URL, delimiter=",")
	print("Dataset Shape: ", data.shape)

	# Split the dataset in as predictors and target
	X, y = data[:, :-1], data[:, -1]

	# Split the dataset for training and testing
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

	print("X_train shape: ", X_train.shape)
	print("X_test shape: ", X_test.shape)
	print("Y_train shape: ", y_train.shape)
	print("Y_test shape: ", y_test.shape)

	model = xbg.XGBClassifier()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(predictions, y_test)
	print(f"Accuracy : {accuracy:.2f}")

	# Saving the model
	with open('model.pkl', 'wb') as f:
		pickle.dump(model, f)

if __name__ == "__main__":
	main()