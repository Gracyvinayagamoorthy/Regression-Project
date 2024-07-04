#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


from sklearn.metrics import mean_squared_error,r2_score


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import seaborn as sns


# In[10]:


df = pd.read_csv("C:\\Users\\aedpu\\Downloads\\water_potability.csv")


# In[11]:


print(df.head())


# In[12]:


print(df.isnull().sum())


# In[13]:


df = df.dropna()


# In[17]:


print(df.head())


# In[22]:


X = df.drop('Hardness',axis = 1)


# In[23]:


Y = df['Hardness']


# In[48]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3 ,random_state = 15)


# In[ ]:





# In[31]:


model = LinearRegression()


# In[32]:


model.fit(X_train,Y_train)


# In[33]:


Y_pred = model.predict(X_test)


# In[34]:


mse = mean_squared_error(Y_test,Y_pred)


# In[35]:


r2 = r2_score(Y_test,Y_pred)


# In[36]:


print(f'Mean Squared Error: {mse}')


# In[37]:


print(f'R-Squared Score:{r2}')


# In[51]:


plt.figure(figsize = (10,6))
plt.scatter(Y_test,Y_pred,color = 'blue')
plt.plot([min(Y_test),max(Y_test)], [min(Y_test),max(Y_test)], color = 'red',linewidth = 2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title("Actual Vs Predicted Water Quality")
plt.show()


# In[ ]:




