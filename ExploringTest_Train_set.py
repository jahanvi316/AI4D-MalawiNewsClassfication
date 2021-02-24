#!/usr/bin/env python
# coding: utf-8

# In[101]:


#pip install -U scikit-learn


# In[102]:


import numpy as np #for numbers and matrix
import pandas as pd 
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize 


# In[103]:


df=pd.read_csv('Train.csv')


# In[104]:


df.head()


# In[105]:


df.isnull().sum()


# In[106]:


len(df) # number of Articles


# In[107]:


df['Label'].value_counts() 


# In[108]:


Lables= df['Label'].values #sperating the lables from the texts and putting it into an array (all the lables )


# In[109]:


Lables


# In[110]:


Articles= df['Text'].values # contains all the text of the articles 


# In[111]:


Articles


# In[112]:


from sklearn.svm import SVC


# In[113]:


#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[114]:


#from sklearn.pipeline import Pipeline


# In[115]:


#pip uninstall scikit-learn


# In[116]:


#pip install scikit-learn==0.13


# In[117]:


#from sklearn.feature_extraction import TfidfVectorizer


# In[118]:


from sklearn.model_selection import KFold #better to use StratifiedKFold for classfication (NEXT)


# In[119]:


kf = KFold(n_splits=5,shuffle=True )


# In[120]:


print(kf)


# In[121]:


for train, test in kf.split(df):
    print("tarining set",train, "testing set", test)


# In[122]:


from sklearn.model_selection import StratifiedKFold
folds= StratifiedKFold(n_splits=7,shuffle=True )# 7 because 10 was too big for the last fold


# In[123]:


folds.get_n_splits(Articles, Lables)


# In[124]:


for train, test in folds.split(Articles, Lables):
    X_train, X_test = Articles[train], Articles[test]
    y_train, y_test = Lables[train], Lables[test]
    print("tarining set",train, "testing set", test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




