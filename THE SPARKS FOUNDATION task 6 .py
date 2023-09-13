#!/usr/bin/env python
# coding: utf-8

# # TSF Internship Task #6: Prediction using Decision Tree Algorithm

# Purpose: The objective of the project is to create a decision tree classifier, so that if we feed new data to the model, it will be able to predict the right class accordingly. So, it is clear that the problem in hand is a classification problem.

# Language Used: Python 3
# 
# Tools: Kaggle, Github
# 
# Approach: First, we will undersatnd the dataset, clean the data and then use data visualization techniques to visually identify the distinctive features and how they are related with the respective class. Then we will create a decision tree model for future prediction purposes.

# In[1]:


# Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# # Understanding the Dataset

# In[3]:


df=pd.read_csv("C:\\Users\\Asus Bq512TS\\Downloads\\Iris.csv")


# In[4]:


df


# In[5]:


# Basic information about the dataset

df.info()


# In[6]:


# Summary statistics of the Dataset

df.describe().T


# In[7]:


# Finding all the unique species in the dataset

unique_species = df[['Id','Species']].groupby('Species').count()
unique_species


# Observation:
# 
# The dataset has three unique species and each species has 50 entries in the dataset.

# In[8]:


##Data Cleaning


# In[9]:


df.drop_duplicates()


# In[10]:


# Searching for null values

df.isnull().sum()


# Obseravtion:
# 
# There is no null values in the dataset.

# In[11]:


# Creating boxplots to show outliers in each numerical column

def boxplot(col):
    sns.boxplot(df[col], color = 'aqua')
    plt.title(col)
    plt.show()
    
for i in list(df.select_dtypes(exclude = ['object']).columns)[0:]:
    boxplot(i)


# Observation:
# 
# Most of the columns don't contain outliers.

# In[12]:


# Scatterplot of the numerical variables

sns.pairplot(df, hue = 'Species', palette = 'colorblind')


# Observation:
# 
# As we can see, the scatter plot of 'PetalLengthCm' and 'PetalWidthCm' provides a good classification visual.

# In[13]:


# Distibutions of the numerical variables:

def distplot(col):
    plt.figure(figsize=(6,4))
    sns.distplot(df[col], color = 'aqua')
    plt.show()
    
for i in list(df.select_dtypes(exclude=['object']).columns)[0:]:
    distplot(i)


# Observation:
# 
# Except for Sepal Width, none of the variables are distributed normally. Since decision tree regressor model does not require the assumption of normality, we can move forward without normalization.

# In[14]:


# Variation of length and width according to species

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df, palette = 'colorblind')
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df, palette = 'colorblind')
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df, palette = 'colorblind')
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df, palette = 'colorblind')


# In[15]:


# Finding correlation

corr = df[df.columns[1:5]].corr()
sns.heatmap(corr, cmap = 'Greens', annot = True)
plt.show()


# Observation:
# 
# The deeper the green colour, the higher the correlation. Some of the variables are higly correlated. However, decision tree model does not get affected by multi-collinearity. So, it we can move forwars.

# 

# # Decision Tree Model

# In[17]:


# Label Encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Species'] = le.fit_transform(df['Species'])


# In[18]:


df


# In[19]:


# Selecting dependent and independent variables

# Dependent Variable
y = df.Species

# Independent Variables
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]


# In[20]:


x.head()


# Train-Test-Split

# In[22]:


# train and test dataset. Splitting train and test data in 80:20 ratio

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[23]:


train_x.head()


# # Our Model

# In[24]:


from sklearn.tree import DecisionTreeRegressor

iris_model = DecisionTreeRegressor(random_state = 42)
iris_model.fit(train_x, train_y)
predictions = iris_model.predict(test_x)


# In[25]:


# Visualizing the decision tree

from sklearn import tree
tree.plot_tree(iris_model)


# In[27]:


# Accuracy of the model

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, predictions)
print("Accuracy Score:", accuracy*100, "%")


# As we can see, the accuracy of the model is very high. So, the model has a great predictive power.

# In[ ]:




