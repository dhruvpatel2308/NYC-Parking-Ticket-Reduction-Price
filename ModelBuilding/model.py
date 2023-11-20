import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# %matplotlib inline

parking_violation_ds = pd.read_csv('/content/Final_Draft.csv')

parking_violation_ds.head()

parking_violation_ds.columns

data = parking_violation_ds.copy()

def time_minute(time_val):
  hour, minute = time_val.split(":")

  time = int(hour) * 60 + int(minute)
  return time

data['time_minute'] = data['Time'].apply(lambda x: time_minute(x))

# Identify the frequencey distribution of a categorical data
def freq_dist(data, col, figsize=(7, 7), min_col=2, threshold=0.8, orient="v"):
  plt.figure(figsize=figsize)

  freq_cnt = dict(data[col].value_counts())

  total_data_len = len(data)
  null_cnt = data[col].isna().sum()
  actual_data_len = total_data_len - null_cnt

  print(f"{null_cnt} values are null")

  percentage_data = actual_data_len/total_data_len

  print(f"{col} covers {percentage_data:.2f}% of data")

  cummulative_data_items = 0
  category_percentage = {}

  for idx, (key, value) in enumerate(freq_cnt.items()):
    cummulative_data_items += value
    category_percentage[key] = (cummulative_data_items / actual_data_len)

    if category_percentage[key] >= threshold and len(category_percentage) >= min_col:
      break

  if orient == "v":
    ax = sns.barplot(x=list(category_percentage.keys()), y=list(category_percentage.values()), orient=orient)
    ax.set_xticklabels(labels=list(category_percentage), rotation=45)
  else:
    ax = sns.barplot(x=list(category_percentage.values()), y=list(category_percentage.keys()), orient=orient)

  ax.set_title("Categories considering")

  for i in ax.containers:
    ax.bar_label(i, fmt='%.2f')
  plt.show()

categorical_columns = list(data.dtypes[data.dtypes == 'object'].keys())

print("Cateogircal_columns")
print(categorical_columns)

not_include_cat_col = ['Time', 'bins']

freq_dist(data, col='License Type', figsize=(7, 5), min_col=3)

license_type_cat = ['COM', 'PAS', 'OMT']
data['license_type'] = data['License Type'].apply(lambda x: x if x in license_type_cat else 'OTH')

print(data['license_type'].value_counts())

freq_dist(data, 'Violation', figsize=(9, 7), min_col=10, orient='h')

# Taking top 11 category based on the graph
violation_cat = list(data['Violation'].value_counts().keys()[:11])

data['violation'] = data['Violation'].apply(lambda x: x if x in violation_cat else 'OTH')
print(data['violation'].value_counts())

freq_dist(data, 'Issuing Agency', figsize=(9, 7), min_col=3)

issuing_agency_cat = list(data['Issuing Agency'].value_counts().keys()[:3])
data['issuing_agency'] = data['Issuing Agency'].apply(lambda x: x if x in issuing_agency_cat else 'OTH')

print(data['issuing_agency'].value_counts())

freq_dist(data, 'County', figsize=(9, 7), min_col=5)

freq_dist(data, 'Violation Status', figsize=(9, 7), min_col=5)

violation_status_cat = list(data['Violation Status'].value_counts().keys()[:5])
data['violation_status'] = data['Violation Status'].apply(lambda x: x if x in violation_status_cat else 'OTH')
print(data['violation_status'].value_counts())

freq_dist(data, 'Violation Precinct', figsize=(9, 7), orient='h')

# There are a total of 74 precinct hence we can select top 20 among them
# that tends to represent the data well
violation_precinct_cat = list(data['Violation Precinct'].value_counts().keys()[:20])
data['violation_precinct'] = data['Violation Precinct'].apply(lambda x: x if x in violation_precinct_cat else 'OTH')
print(data['violation_precinct'].value_counts())

"""## Model Implementation"""

new_cat_columns = ['license_type', 'violation', 'issuing_agency', 'County', 'Law Section', 'DayOfWeek', 'violation_precinct']
features = ['Fine Amount', 'Penalty Amount', 'Interest Amount', 'Reduction Amount', 'time_minute']

features += new_cat_columns

new_cat_columns

dataset = data[features]
dtype_dict_cat = {}
for col in new_cat_columns:
  dtype_dict_cat[col] = 'category'

dataset = dataset.astype(dtype_dict_cat)

data_encoded = pd.get_dummies(dataset, columns=new_cat_columns)
print(data_encoded.head())

y = data_encoded['Reduction Amount']
X = data_encoded.drop('Reduction Amount', axis=1)

print("X = ", X.shape)
print("Y = ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state = 13
)

print("Train Dataset shape: ", X_train.shape, " | y_train: ", y_train.shape)
print("Test Dataset shape: ", X_test.shape, " | y_test: ", y_test.shape)

"""## Working out on model"""

model = XGBRegressor(max_depth=10)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=13)
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

print(f"Mean MAE: {scores.mean():.2f} | Std: {scores.std():.2f}")

model.fit(X_train, y_train)

def mean_square_error(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  result = np.mean((y_true - y_pred) ** 2)
  return result

# model = XGBRegressor(max_depth=15, learning_rate=0.3, n_estimators=1000, objective='reg:squarederror')
model = XGBRegressor(max_depth=10, n_estimators=500, objective='reg:absoluteerror', eval_metric='mae')

model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

print(f"Mean Square Error: {mean_square_error(y_train, y_pred_train):.4}")
print(f"Mean Square Error: {mean_square_error(y_test, y_pred_test):.4}")

print(f"Model score: {model.score(X_test, y_test):.4f}")

y_pred_train = model.predict(X_train)
print(f"Mean Square Error Train: {mean_square_error(y_train, y_pred_train)}")

"""# Model Tuning"""

"""

1.   Fit the parameters to scale
2.   Transform them
3.   Stack them back to their original place

"""

dataset.columns

scaler = StandardScaler()
scaler.fit(dataset[['Fine Amount', 'Penalty Amount', 'Interest Amount', 'Reduction Amount', 'time_minute']])

scaled_data = scaler.transform(dataset[['Fine Amount', 'Penalty Amount', 'Interest Amount', 'Reduction Amount', 'time_minute']])

print(scaled_data.shape)

dataset['Fine Amount'] = scaled_data[:, 0]
dataset['Penalty Amount'] = scaled_data[:, 1]
dataset['Interest Amount'] = scaled_data[:, 2]
dataset['Reduction Amount'] = scaled_data[:, 3]
dataset['time_minute'] = scaled_data[:, 4]

dataset.head()

data_encoded = pd.get_dummies(dataset, columns=new_cat_columns)
data_encoded.head()

y = data_encoded['Reduction Amount']
X = data_encoded.drop('Reduction Amount', axis=1)

print("X = ", X.shape)
print("Y = ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state = 13
)

print("Train Dataset shape: ", X_train.shape, " | y_train: ", y_train.shape)
print("Test Dataset shape: ", X_test.shape, " | y_test: ", y_test.shape)

# model = XGBRegressor(max_depth=15, learning_rate=0.3, n_estimators=1000, objective='reg:squarederror')
model = XGBRegressor(
    max_depth=10,
    n_estimators=750,
    learning_rate=0.3,
    objective='reg:absoluteerror',
    eval_metric='mae'
  )

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=True

)
# y_pred_test = model.predict(X_test)
# y_pred_train = model.predict(X_train)

# print(f"Mean Square Error: {mean_square_error(y_train, y_pred_train):.4}")
# print(f"Mean Square Error: {mean_square_error(y_test, y_pred_test):.4}")

print(f"Model score: {model.score(X_test, y_test):.4f}")

"""
Tree dict
max_depth n_est lr   score_train score_test
15        500   0.3  0.21808     0.5596
10        500   0.3  0.19338     0.6057
10        750   0.3  0.19338     0.6057
"""