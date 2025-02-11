#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


df = pd.read_csv("C:/Users/harsh/Downloads/crops/yield_df.csv")


# In[3]:


df.head(10)


# In[4]:


df["Area"].unique()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df = df.drop(['Unnamed: 0'] , axis = 1)


# In[8]:


#in avg rinfall there are some values which are numbers but are in string formate so we need to remove that 
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True


# In[9]:


StringObj= df[df['average_rain_fall_mm_per_year'].apply(isStr)].index


# In[10]:


df = df.drop(StringObj)


# In[11]:


df.head(2)


# In[12]:


country_yeild= []
country = df['Area'].unique()
for i in country:
    country_yeild.append(df[df["Area"]==i]['hg/ha_yield'].sum())


# In[13]:


country


# In[14]:


country_yeild


# In[15]:


total_yeild = df['hg/ha_yield'].sum()
total_yeild


# In[16]:


#frequency plot 
plt.figure(figsize=(10,20))
sns.barplot(y=country , x=country_yeild)


# In[17]:


#particular crop grown by the country
df['Item'].value_counts()


# In[18]:


items = df['Item'].unique()
items


# In[19]:


country_item = []
for i in items:
    country_item.append(df[df["Item"]==i]['hg/ha_yield'].sum())
    


# In[20]:


len(country_item)


# In[21]:


plt.figure(figsize=(20,5))
sns.barplot(y=country_item , x = items)


# In[22]:


#train test split 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[23]:


df.head()


# In[24]:


x = df.drop(['hg/ha_yield'] , axis = 1)
y = df['hg/ha_yield']


# In[25]:


x_train , x_test , y_train ,y_test = train_test_split(x,y ,test_size=0.2 , random_state=40)


# In[26]:


#handeling catagorical data
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer


# In[27]:


ohe = OneHotEncoder(drop="first")
scaler = StandardScaler()


# In[28]:


x_train


# In[60]:


t1 = ColumnTransformer(
transformers=[
    ('ohe',ohe,['Area', 'Item']),
    ('Standarization', scaler , ['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp'])

],
remainder='passthrough'

)


# In[61]:


t1


# In[62]:


x_train_dummy = t1.fit_transform(x_train)
x_test_dummy = t1.fit_transform(x_test)


# In[63]:


x_test_dummy


# In[64]:


from sklearn.linear_model import Lasso,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score # using r2_score since we are working with regression model not classification model


# In[65]:


models = {
    'lr':LinearRegression(),
    'lasso':Lasso(),
    'KNN':KNeighborsRegressor(),
    'DTC':DecisionTreeRegressor()
}

for name , model in models.items():
    model.fit(x_train_dummy,y_train)
    pred = model.predict(x_test_dummy)
    acc = r2_score(y_test, pred)*100  # here used accuacy_score but got error since its used for classification thing not regression thing
    
    print(f"{name} accuracy :{acc}%")


# In[66]:


knn  = KNeighborsRegressor()
knn.fit(x_train_dummy,y_train)


# In[67]:


knn.predict(x_test_dummy)


# In[77]:


def prediction(Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item):
    features = pd.DataFrame([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],columns=['Area', 'Item', 'Year', 'average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp'])
    features_trasnformer = t1.transform(features)
    pred = knn.predict(features_trasnformer).reshape(1,-1)
    return pred[0]


# In[78]:


x_train.head(1)


# In[79]:


Area ='India'
Item = 'Rice, paddy'
Year = 1990
average_rain_fall_mm_per_year = 1208.3
pesticides_tonnes = 100000.34
avg_temp = 30.0

res =prediction(Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp)


# In[80]:


res[0]


# In[84]:


import pickle
pickle.dump(knn , open("C:/Users/harsh/Desktop/flask/knn model.pkl" , 'wb'))
pickle.dump(t1 , open("C:/Users/harsh/Desktop/flask/tranform.pkl" , 'wb'))


# In[85]:


import sklearn
print(sklearn.__version__)


# In[83]:


x_train.columns


# In[86]:


df.head()


# In[5]:


n = 2
arr= [1,2,4,4,5]
for i in arr:
    if n!=i:
        break
        


# In[ ]:




