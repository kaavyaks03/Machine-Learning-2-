#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset=pd.read_csv("mobile data set.csv")


# In[3]:


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(x)
print(y)


# In[4]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[5]:


from sklearn.preprocessing import StandardScaler
s=StandardScaler()
x_train=s.fit_transform(x_train)
x_test=s.transform(x_test)


# In[6]:


from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)


# In[7]:


classifier.fit(x_train,y_train)


# In[8]:


y_ans=classifier.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
z=accuracy_score(y_test,y_ans)
print(z)
z1=confusion_matrix(y_test,y_ans)
print(z1)


# In[9]:


y_pred=classifier.predict(x_train)
a=accuracy_score(y_train,y_pred)
print(a)
a1=confusion_matrix(y_train,y_pred)
print(a1)

