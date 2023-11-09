#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)



import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import ( 
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    classification_report
)


# In[2]:


data = pd.read_csv("parking_violation_geocoded.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.columns


# In[6]:


data = data.drop(["summons_image"],axis=1)


# In[7]:


data.info()


# In[8]:


data.describe().T


# In[9]:


data['issue_date'] = pd.to_datetime(data['issue_date'], errors='coerce')
data['issue_date'] = data['issue_date'].dt.strftime('%Y-%m-%d')


# In[10]:


for i in data.columns :
    print(i, " : ",data[i].isna().sum())
    print("*" * 50)


# In[11]:


data = data.drop(["violation_description","vehicle_year","violation_location","issuer_command","issuer_squad","unregistered_vehicle","meter_number","violation_legal_code","violation_status","violation_post_code"],axis=1)
data.shape


# In[12]:


data.info()


# In[13]:


data.eq(0).sum()
for i in data.columns:
    if(data[i].isna().sum() > 0 ) : 
        print(i, ": ",data[i].isna().sum())
        print("*" * 50)


# In[14]:


data = data.dropna()


# In[15]:


data.info()


# In[16]:


data.describe().T


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




