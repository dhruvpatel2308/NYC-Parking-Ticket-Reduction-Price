#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import joblib
from sklearn.model_selection import GridSearchCV
from datetime import datetime

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_curve,
    roc_curve,
    mean_squared_error
)


# In[2]:


data = pd.read_csv("merged_file.csv")


# In[3]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[4]:


data.head()


# In[5]:


data = data.drop(["Summons Image"],axis=1)
data['Issue Date'] = pd.to_datetime(data['Issue Date'], errors='coerce')
data['Issue Date'] = data['Issue Date'].dt.strftime('%Y-%m-%d')


# In[6]:


data.columns


# In[7]:


data = data.drop(["Summons Number","Violation Status","Issuer Code"],axis=1)


# In[8]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[9]:


data = data.drop(["Precinct"],axis=1)


# In[10]:


data = data.drop(["Plate","State"],axis=1)


# In[11]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[12]:


data.dropna(subset=['Issuing Agency'], inplace=True)


# In[13]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[14]:


borough_to_county = {
    'NY': 'New York',
    'BX': 'Bronx',
    'BK': 'Kings',
    'QN': 'Queens',
    'ST': 'Richmond',
    'Q': 'Queens',
    'QN': 'Queens',
    'K': 'Kings',
    'R': 'Richmond',
    'Kings': 'Kings',
    'Bronx': 'Bronx',
    'Qns': 'Queens',
    'Rich': 'Richmond',
    'QUEEN': 'Queens',
    'QNS': 'Queens',
    'MN' : 'New York'
}


# In[15]:


data['County'] = data['County'].replace(borough_to_county)


# In[16]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[17]:


data.dropna(subset=['County','Violation'], inplace=True)


# In[18]:


data.shape


# In[19]:


violation_data = pd.read_excel("ParkingViolationCodes_January2020.xlsx")
violation_data.head()
violation_data.columns


# In[20]:


data = data.merge(violation_data, left_on='Violation',  right_on="VIOLATION DESCRIPTION", how="left")
data["Actual_Fine_Amount"] = data.apply(
   lambda row: row.loc['Manhattan  96th St. & below\n(Fine Amount $)'] if row['County'] == 'New York' else row.loc['All Other Areas\n(Fine Amount $)'],
    axis=1
)


# In[21]:


data.head()


# In[22]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[23]:


law_mapping ={
    408 : 'Parking, Stopping, Standing',
    1180 : 'Speed Not Reasonable & Prudent',
    1111 : 'Red Light',
    410  : 'Buses'
}


# In[24]:


data["Law Section"] = data["Law Section"].map(law_mapping)


# In[25]:


allowed_license_types = ["PAS", "OMT", "COM"]
data = data[data['License Type'].isin(allowed_license_types)]

allowed_issuing_agency = ["TRAFFIC", "DEPARTMENT OF TRANSPORTATION","DEPARTMENT OF SANITATION","POLICE DEPARTMENT"]

data = data[data['Issuing Agency'].isin(allowed_issuing_agency)]


# In[26]:


violation_precint = data['Violation Precinct'].unique()
issuer_precint = data['Issuer Precinct'].unique()
precints = np.concatenate((violation_precint, issuer_precint))
precints = np.unique(precints)
precints.sort()


# In[27]:


precint_mapping = { num : f"{num} Precinct" for num in precints } 


# In[28]:


data["Violation Precinct"] = data["Violation Precinct"].map(precint_mapping)
data["Issuer Precinct"] = data["Issuer Precinct"].map(precint_mapping)


# In[29]:


data.head()


# In[30]:


data["Is_Different"] = (data["Violation Precinct"] != data["Issuer Precinct"])      


# In[31]:


data.head()


# In[ ]:





# In[32]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[33]:


data.dropna(subset=['VIOLATION CODE'], inplace=True)


# In[34]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[35]:


data.columns


# In[36]:


data.head()


# In[37]:


data = data.drop(["VIOLATION DESCRIPTION","VIOLATION CODE","Judgment Entry Date","Manhattan  96th St. & below\n(Fine Amount $)","All Other Areas\n(Fine Amount $)"],axis=1)


# In[38]:


data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


data["Is_High"] = data["Fine Amount"] > data["Actual_Fine_Amount"]


# In[40]:


data["Total_Amont"] = data["Fine Amount"] + data["Penalty Amount"] + data["Interest Amount"]


# In[ ]:





# In[41]:


data['reduced_fine_amount'] = data['Reduction Amount'] / data['Total_Amont'] * 100


# In[42]:


data = data.drop(["Total_Amont"],axis=1)


# In[43]:


def convert_to_24_hour_format(time_12_hour):
    if(time_12_hour == "nan" or isinstance(time_12_hour, float)) :
        return "00:00"

    time_parts = time_12_hour.split(':')
    if('.' in time_parts[0]) :
        hours = 0
    else :
        hours = int(time_parts[0])
    minutes_period = time_parts[1].split()
    period = minutes_period[0][-1].upper()  # Convert to uppercase for case insensitivity

    if period == 'P' and hours < 12:
        hours += 12
    elif period == 'A' and hours == 12:
        hours = 0
    hours_str = str(hours).zfill(2)
    time_24_hour = f"{hours_str}:00"
    return time_24_hour


# In[44]:


data['Time'] = data['Violation Time'].apply(convert_to_24_hour_format)


# In[45]:


data = data.drop(["Violation Time"],axis=1)


# In[46]:


data['issue_date_time'] = pd.to_datetime(data['Issue Date'])

if pd.api.types.is_datetime64_any_dtype(data['issue_date_time']):
    data['DayOfWeek'] = data['issue_date_time'].dt.day_name()
else:
    print("The 'issue_date' column is not in a datetime format.")


# In[47]:


data['month'] = data['issue_date_time'].dt.month


# In[48]:


data = data.drop(["issue_date_time"],axis=1)


# In[49]:


data = data.drop(["Issue Date"],axis=1)


# In[50]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[51]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[52]:


data.shape


# In[53]:


data.info()


# In[54]:


# data_copy = data.copy()

# allowed_license_types = ["PAS", "OMT", "COM"]
# data_copy = data_copy[data_copy['License Type'].isin(allowed_license_types)]

# allowed_issuing_agency = ["TRAFFIC", "DEPARTMENT OF TRANSPORTATION","DEPARTMENT OF SANITATION","POLICE DEPARTMENT"]

# data_copy = data_copy[data_copy['Issuing Agency'].isin(allowed_issuing_agency)]


# In[55]:


data.shape


# In[56]:


bins = [0, 11, 21,31,41,51,61,71,81,91,101,500]
bin_labels = ['0-10', '10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100','>100']
# Create the bins using pd.cut
data['Reduction_Per'] = pd.cut(data['reduced_fine_amount'], bins=bins, labels=bin_labels,right=False)


# In[ ]:





# In[ ]:





# In[57]:


# precinct_mapping = {
#     0: "Unknown Precinct",
#     1: "1st Precinct",
#     5: "5th Precinct",
#     6: "6th Precinct",
#     7: "7th Precinct",
#     9: "9th Precinct",
#     10: "10th Precinct",
#     13: "13th Precinct",
#     17: "17th Precinct",
#     19: "19th Precinct",
#     20: "20th Precinct",
#     23: "23rd Precinct",
#     24: "24th Precinct",
#     25: "25th Precinct",
#     26: "26th Precinct",
#     28: "28th Precinct",
#     30: "30th Precinct",
#     32: "32nd Precinct",
#     33: "33rd Precinct",
#     40: "40th Precinct",
#     41: "41st Precinct",
#     42: "42nd Precinct",
#     43: "43rd Precinct",
#     44: "44th Precinct",
#     45: "45th Precinct",
#     46: "46th Precinct",
#     47: "47th Precinct",
#     48: "48th Precinct",
#     49: "49th Precinct",
#     50: "50th Precinct",
#     52: "52nd Precinct",
#     60: "60th Precinct",
#     61: "61st Precinct",
#     62: "62nd Precinct",
#     63: "63rd Precinct",
#     66: "66th Precinct",
#     67: "67th Precinct",
#     68: "68th Precinct",
#     69: "69th Precinct",
#     70: "70th Precinct",
#     71: "71st Precinct",
#     72: "72nd Precinct",
#     73: "73rd Precinct",
#     75: "75th Precinct",
#     76: "76th Precinct",
#     77: "77th Precinct",
#     78: "78th Precinct",
#     79: "79th Precinct",
#     81: "81st Precinct",
#     83: "83rd Precinct",
#     84: "84th Precinct",
#     88: "88th Precinct",
#     90: "90th Precinct",
#     94: "94th Precinct",
#     100: "100th Precinct",
#     101: "101st Precinct",
#     102: "102nd Precinct",
#     103: "103rd Precinct",
#     104: "104th Precinct",
#     105: "105th Precinct",
#     106: "106th Precinct",
#     107: "107th Precinct",
#     108: "108th Precinct",
#     109: "109th Precinct",
#     110: "110th Precinct",
#     111: "111th Precinct",
#     112: "112th Precinct",
#     113: "113th Precinct",
#     114: "114th Precinct",
#     115: "115th Precinct",
#     120: "120th Precinct",
#     121: "121st Precinct",
#     122: "122nd Precinct",
#     123: "123rd Precinct"
# }


# In[58]:


# data_copy["Violation Precinct"] = data_copy["Violation Precinct"].map(precinct_mapping)


# In[59]:


# data_copy["Issuer Precinct"] = data_copy["Issuer Precinct"].map(precinct_mapping)


# In[60]:


# data_copy["is_Different"] = (data_copy["Violation Precinct"] != data_copy["Issuer Precinct"])                             


# In[61]:


# data_copy = data_copy.drop(["is_Different"],axis=1)


# In[62]:


# law_mapping ={
#     408 : 'Parking, Stopping, Standing',
#     1180 : 'Speed Not Reasonable & Prudent',
#     1111 : 'Red Light',
#     410  : 'Buses'
# }


# In[63]:


# data_copy["Law Section"] = data_copy["Law Section"].map(law_mapping)


# In[64]:


# data_copy.head()


# In[65]:


# nan_rows = data_copy[data_copy["Actual_Fine_Amount"].isna() ]
# nan_rows


# In[ ]:





# In[ ]:





# In[66]:


data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


data = data.drop(["Actual_Fine_Amount","Issuer Squad","reduced_fine_amount"],axis=1)


# In[68]:


data.head()


# In[69]:


for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[70]:


data.shape


# In[71]:


data_copy = data.copy()


# In[72]:


data_copy = data_copy.dropna()


# In[73]:


data.shape


# In[74]:


data_copy.shape


# In[75]:


data_copy.head()


# In[ ]:





# In[ ]:





# In[76]:


# data.to_csv("DataFinal",index=False)


# In[ ]:





# In[77]:


cat_columns = data_copy.describe(include=["object"]).columns 

data_encoded = pd.get_dummies(data_copy, columns=cat_columns)
data_encoded.head()


# In[78]:


# df = pd.DataFrame(0, columns=data_encoded.columns, index=[0])

# # Specify the output CSV file name
# output_file = 'output.csv'

# # Export the DataFrame to a CSV file
# df.to_csv(output_file, index=False)


# In[ ]:





# In[ ]:





# In[79]:


X = data_encoded.drop(['Reduction Amount','Reduction_Per'], axis=1)  # Features
y = data_encoded['Reduction_Per']  # Target variable


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[84]:


# rf_classifier = RandomForestClassifier(random_state=42)


# In[85]:


# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }


# In[86]:


# grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)


# In[84]:


# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_


# In[85]:


# y_pred = best_model.predict(X_test)


# In[86]:


# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)


# In[87]:


# # Print the results
# print("Best Parameters:", best_params)
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")
# print(f"Confusion Matrix:\n{conf_matrix}")

# # Get feature importances from the best model
# feature_importances = best_model.feature_importances_

# # Sort indices by importance in descending order
# sorted_indices = np.argsort(feature_importances)[::-1]

# # Print feature ranking
# print("Feature ranking:")
# for f in range(X.shape[1]):
#     print(f"{X.columns[sorted_indices[f]]} ({feature_importances[sorted_indices[f]]})")


# In[87]:


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Visualize one of the decision trees in the Random Forest
estimator = clf.estimators_[0]
tree_rules = export_text(estimator, feature_names=X.columns.tolist())
print(tree_rules)

# Visualize feature importances
feature_importances = clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(X.columns, feature_importances)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importances in Random Forest Classifier')
plt.xticks(rotation=45)
plt.show()


# In[88]:


joblib.dump(clf, "best.sav.gz")


# In[89]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')
# roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovo')
conf_matrix = confusion_matrix(y_test, y_pred)


# In[90]:


# Print the results
# print("Best Parameters:", best_params)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Get feature importances from the best model
# feature_importances = best_model.feature_importances_

# # Sort indices by importance in descending order
# sorted_indices = np.argsort(feature_importances)[::-1]

# # Print feature ranking
# print("Feature ranking:")
# for f in range(X.shape[1]):
#     print(f"{X.columns[sorted_indices[f]]} ({feature_importances[sorted_indices[f]]})")


# In[91]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Replace the given matrix with your confusion matrix
conf_matrix = np.array([
    [32279, 0, 0, 0, 0, 0, 0, 0, 0, 5],
    [30, 17769, 1, 0, 0, 2, 0, 0, 0, 9],
    [22, 0, 49536, 1, 0, 0, 0, 0, 0, 19],
    [58, 4, 4, 4855, 1, 0, 1, 0, 0, 10],
    [77, 0, 5, 0, 1358, 0, 0, 0, 0, 6],
    [3, 0, 0, 2, 0, 3306, 6, 0, 0, 9],
    [16, 0, 0, 0, 0, 0, 12329, 0, 0, 12],
    [5, 0, 1, 0, 0, 0, 0, 29, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [60, 3, 1, 0, 0, 0, 5, 0, 0, 53761]
])

# Calculate the percentage values for better visualization (optional)
conf_matrix_percentage = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Plot the confusion matrix using seaborn heatmap
sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=np.arange(1, 11), yticklabels=np.arange(1, 11))

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




