#!/usr/bin/env python
# coding: utf-8

# # Name :- Banti kumar
# Email :- bantikumar.netb@gmail.com    

# # Task 1 : Prediction using Supervised Machine Learning
#     
# GRIP @ The Sparks Foundation
# 
# In this regression task I tried to predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# 
# This is a simple linear regression task as it involves just two variables.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Load

# In[2]:


#data = pd.read_csv('student_info.csv')
# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(10)


# In[3]:


data.tail()


# In[4]:


data.isnull().sum().head()
# True = null
# False = Not NUll


# In[5]:


data.shape


# # Data Discover and Visualization

# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


plt.scatter(x = data.Hours , y = data.Scores)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Scatter plot of student Scores vs  student hours')
plt.show()


# In[9]:


#Another way to plot graph
data.plot(kind='scatter', x='Hours', y='Scores',alpha=1)


# In[10]:


from pandas.plotting import scatter_matrix
attributes = ['Hours' , 'Scores']
scatter_matrix(data[attributes] , )


# # Fill Missing Attributes

# In[11]:


data.mean()
data1 = data.fillna(data.mean())
#data1.describe()
data1.isnull().sum()


# # looking for correlation

# In[12]:


corr_matrix = data.corr()
corr_matrix['Scores'].sort_values(ascending=False)


# # Train Test Split

# In[13]:


X = data1.drop('Scores' , axis = "columns")
y = data1.drop('Hours' , axis ='columns')
print(f"shape of x is {X.shape} \nShape of y is {y.shape}")


# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y , test_size = 0.2 , random_state = 51)
print('x_train = ',len(x_train))
print('x_test = ',len(x_test))
print('y_train = ',len(y_train))
print('y_test = ',len(y_test))


# # Build a Machine Learning Model

# In[15]:


# y = m*x+c
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()


# In[16]:


lr_model.fit(x_train, y_train)
x_pre = lr_model.predict(x_test)
lr_model.score(x_test,y_test)


# In[17]:


dataframe = pd.DataFrame(np.c_[x_test, y_test,x_pre], columns = ['study_hours','Score','Score_predicted'])
dataframe


# # Fine Tune your Model

# In[18]:


plt.scatter(x_train,y_train)


# In[19]:


plt.scatter(x_test, y_test)
plt.plot(x_train, lr_model.predict(x_train), color = 'r')


# # Save your Model

# In[20]:


import joblib
joblib.dump(lr_model ,'Student_marks_predictor_model.pkl')


# In[21]:


model = joblib.load('Student_marks_predictor_model.pkl')
model


# In[22]:


model.predict([[5]])


# # Conclusion
# 
# I was successfully able to carry-out Prediction using Supervised ML task 
# and was able to evaluate the model's performance on various parameters.

# # Thank You
