
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sklearn

from sklearn import tree
from pandas import *

df = pd.read_csv("movie_metadata.csv")
df =df.dropna()


# In[2]:

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=1)


# In[3]:

from sklearn.preprocessing import LabelEncoder
X = pd.DataFrame()
df = pd.read_csv("movie_metadata.csv")
df = df.dropna()


columnsToEncode = list(df.select_dtypes(include=['category','object']))
le = LabelEncoder()
for feature in columnsToEncode:
    try:
        df[feature] = le.fit_transform(df[feature])
    except:
        print('Error encoding ' + feature)
df.head()


# In[4]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr, vmax=1, square=True)


# In[5]:

X=df
y=X['imdb_score']
#y.apply(np.round)
X = X.drop(['imdb_score'], axis = 1)

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X = scaler.fit_transform(X)
y = np.array(y).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state =190)


# ## Linear and Logistic Regression

# In[6]:

from sklearn.linear_model import LogisticRegression
LRL2 = LogisticRegression(penalty = 'l2')
LRL2.fit(X_train,y_train)
L2score = LRL2.score(X_test,y_test)
print (L2score)


# In[7]:

from sklearn.linear_model import LinearRegression 
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=4, normalize=True)
model.fit(X_train,y_train)
print('Accuracy: ', model.score(X_test, y_test))


# ## Decision Tree

# In[8]:

#predict and score
from sklearn import tree
tree_model = tree.DecisionTreeClassifier(max_depth = 11, min_samples_split=90)
tree_model.fit(X_train, y_train)
tree_model.score(X_test, y_test)


# ## SVM

# In[9]:

from sklearn import svm

clf = svm.SVC()
clf.set_params(C=10)
clf.fit(X_train, y_train)  
clf.predict(X_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
prediction = clf.predict(X_test)
print(accuracy_score(y_test,prediction))


# ## Linear SVM

# In[10]:

clfLin = svm.SVC(kernel = 'linear')
clfLin.set_params(C=0.5)
clfLin.fit(X_train,y_train)
clfLin.predict(X_test)

from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
prediction = clfLin.predict(X_test)
print(accuracy_score(y_test,prediction))


# ## KNN

# In[11]:

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=12)

neigh.fit(X_train, y_train) 
neigh.set_params(p=7)
neigh.predict(X_test)
accuracy_score(y_test, neigh.predict(X_test))


# ## Random Forest

# In[13]:

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=28)
clf = clf.fit(X_train, y_train)
accuracy_score(y_test,clf.predict(X_test))



