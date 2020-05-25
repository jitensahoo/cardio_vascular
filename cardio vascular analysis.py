#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cardio=pd.read_csv('D:/CardioVascularDisease/CardioVascularDisease/cardio_train.csv',sep=';')
cardio.head()


# In[104]:


cardio.drop_duplicates(inplace=True)


# In[3]:


cardio['age']=cardio['age'].apply(lambda x : x//365)
cardio.head()


# In[4]:


cardio['BMI']=cardio['weight']//(cardio['height']/100)**2


# In[5]:


cardio1.drop(cardio1[cardio1['ap_hi'] < cardio1['ap_lo']].index,axis=0,inplace=True)


# In[35]:


cardio1[cardio1['BMI'] > 35]


# In[6]:


q1=cardio['BMI'].quantile(0.25)
q2=cardio['BMI'].quantile(0.75)
iqr=(q1+q2)/2
q2 + (1.5*iqr)


# In[9]:


#cardio[(cardio['BMI'] < (q1 -(1.5*iqr))) | (cardio['BMI'] > (q2 + (1.5*iqr))]
len(cardio)


# In[7]:


cardio1=cardio[~(cardio['BMI'] > 40) | (cardio['BMI'] < 15)]


# In[13]:


cardio1.drop(cardio1[(cardio1['ap_hi'] > 240) | (cardio1['ap_hi'] < 90) | (cardio1['ap_lo'] < 60) | (cardio1['ap_lo'] > 120)].index,axis=0,inplace=True)


# In[15]:


cardio1[(cardio1['ap_hi'] < 80) | (cardio1['ap_hi'] > 250)]
#cardio1[(cardio1['ap_lo'] < 60) | (cardio1['ap_lo'] > 140)]


# In[16]:


cardio1.drop(cardio1[cardio1['BMI'] < 15 ].index,axis=0,inplace=True)


# In[27]:


cardio1


# In[28]:


from sklearn.feature_selection import SelectKBest,chi2


# In[36]:


sel=SelectKBest(score_func=chi2,k=10)
sel_new=sel.fit_transform(cardio1.iloc[:,1:12],cardio1.iloc[:,13])


# In[37]:


df_score=pd.DataFrame(sel.scores_)
df_col=pd.DataFrame(cardio1.columns)
feature_score=pd.concat([df_col,df_score],axis=1)
feature_score.columns=['attr','score']
feature_score.nlargest(10,'score')


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[41]:


x=cardio1.iloc[:,[1,3,4,5,6,7,8,9,10]]
y=cardio1['cardio']


# In[42]:


x.head()


# In[43]:


lr=LogisticRegression()
cv=cross_val_score(lr,x,y,cv=5)
print(cv)
print(cv.mean())


# In[44]:


#lr=LogisticRegression()
#rfc=RandomForestClassifier()
abc=AdaBoostClassifier(n_estimators=100,learning_rate=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=234,test_size=0.3)
abc.fit(x_train,y_train)


# In[46]:


confusion_matrix(y_test,abc.predict(x_test))
accuracy_score(y_test,abc.predict(x_test))


# In[66]:


classification_report(abc.predict(x_test),y_test)

