#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


# In[3]:


df = pd.read_csv('C:\\Users\shashikant\Desktop\polynomial_regression\polynomial.csv')
df


# In[4]:


x = df[['level']].values
x


# In[7]:


y = df[['salary']].values
y


# In[8]:


model = DecisionTreeRegressor()


# In[9]:


model.fit(x,y)


# In[10]:


model.predict([[6.5]])


# In[19]:


x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.title('Decision tree Algorithm')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.scatter(x,y,color = 'g')
plt.plot(x_grid,model.predict(x_grid),color = 'b')


# In[ ]:




