#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train_data = pd.read_csv("C:/Users/Lenovo/Desktop/Machine Learnig/9.Day_ProjectExplenation/training_set.csv")
test_data = pd.read_csv("C:/Users/Lenovo/Desktop/Machine Learnig/9.Day_ProjectExplenation/testing_set.csv")


# In[2]:


train_data.head()


# In[3]:


test_data.head()


# In[4]:


train_data.isna().sum()


# In[5]:


train_data.Alley = train_data.Alley.fillna("No alley access")
train_data.BsmtQual = train_data.BsmtQual.fillna("No Basement")
train_data.BsmtCond = train_data.BsmtCond.fillna("No Basement")
train_data.BsmtExposure = train_data.BsmtExposure.fillna("No Basement")
train_data.BsmtFinType1 = train_data.BsmtFinType1.fillna("No Basement")
train_data.BsmtFinType2 = train_data.BsmtFinType2.fillna("No Basement")
train_data.FireplaceQu = train_data.FireplaceQu.fillna("No Fireplace")
train_data.GarageType = train_data.GarageType.fillna("No Garage")
train_data.GarageFinish = train_data.GarageFinish.fillna("No Garage")
train_data.GarageQual = train_data.GarageQual.fillna("No Garage")
train_data.GarageCond = train_data.GarageCond.fillna("No Garage")
train_data.PoolQC = train_data.PoolQC.fillna("No Pool")
train_data.Fence = train_data.Fence.fillna("No Fence")
train_data.MiscFeature = train_data.MiscFeature.fillna("None")

test_data.Alley = test_data.Alley.fillna("No alley access")
test_data.BsmtQual = test_data.BsmtQual.fillna("No Basement")
test_data.BsmtCond = test_data.BsmtCond.fillna("No Basement")
test_data.BsmtExposure = test_data.BsmtExposure.fillna("No Basement")
test_data.BsmtFinType1 = test_data.BsmtFinType1.fillna("No Basement")
test_data.BsmtFinType2 = test_data.BsmtFinType2.fillna("No Basement")
test_data.FireplaceQu = test_data.FireplaceQu.fillna("No Fireplace")
test_data.GarageType = test_data.GarageType.fillna("No Garage")
test_data.GarageFinish = test_data.GarageFinish.fillna("No Garage")
test_data.GarageQual = test_data.GarageQual.fillna("No Garage")
test_data.GarageCond = test_data.GarageCond.fillna("No Garage")
test_data.PoolQC = test_data.PoolQC.fillna("No Pool")
test_data.Fence = test_data.Fence.fillna("No Fence")
test_data.MiscFeature = test_data.MiscFeature.fillna("None")


# In[6]:


train_data.isna().sum()


# In[7]:


test_data.isna().sum()


# In[26]:


from module import replacer
replacer(train_data)
replacer(test_data)


# # EDA

# In[29]:


cat = []
con = []
for i in train_data.columns:
    if(train_data[i].dtypes == "object"):
        cat.append(i)
    else:
        con.append(i)


# In[30]:


cat


# In[31]:


con


# In[33]:


train_data.corr()["SalePrice"].sort_values()


# # Define X and Y

# In[35]:


train_data.isna().sum()


# In[37]:


X = train_data.drop(labels=["Id","SalePrice"],axis=1)
Y = train_data[["SalePrice"]]


# In[38]:


X.isna().sum()


# # Remove Outliers

# In[39]:


from module import standardize,outliers
X1 = standardize(X)
OL = outliers(X1)


# In[40]:


X = X.drop(index=OL,axis=0)
Y = Y.drop(index=OL,axis=0)


# In[41]:


X.shape


# In[42]:


X.index = range(0,1021,1)
Y.index = range(0,1021,1)


# # Preprocessing

# In[43]:


from module import preprocessing
Xnew = preprocessing(X)


# In[45]:


Xnew.isna().sum()


# # Training testing split

# In[46]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# # Create a Backward elemination OLS model

# In[47]:


from statsmodels.api import OLS,add_constant
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst)
model = ol.fit()
rsq = model.rsquared_adj
col_to_drop = model.pvalues.sort_values().index[-1]
print("Dropped: column:",col_to_drop,"\tRsquared:",round(rsq,4))
Xnew = Xnew.drop(labels=col_to_drop,axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)
xconst = add_constant(xtrain)
ol = OLS(ytrain,xconst)
model = ol.fit()
rsq = model.rsquared_adj


# In[48]:


for i in range(0,10):
    from statsmodels.api import OLS,add_constant
    xconst = add_constant(xtrain)
    ol = OLS(ytrain,xconst)
    model = ol.fit()
    rsq = model.rsquared_adj
    col_to_drop = model.pvalues.sort_values().index[-1]
    print("Dropped: column:",col_to_drop,"\tRsquared:",round(rsq,4))
    Xnew = Xnew.drop(labels=col_to_drop,axis=1)
    xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)
    xconst = add_constant(xtrain)
    ol = OLS(ytrain,xconst)
    model = ol.fit()
    rsq = model.rsquared_adj


# # Create a Linear model based on selected features

# In[49]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err = mean_absolute_error(ytest,ts_pred)


# In[50]:


tr_err


# In[51]:


ts_err


# In[53]:


#Xnew.corr()


# # Regularize

# In[54]:


lambdas = []
q = 8
for i in range(0,4000,1):
    q = q + 0.001
    q = round(q,4)
    lambdas.append(q)


# In[56]:


from sklearn.linear_model import Ridge
tr = []
ts = []
for i in lambdas:
    rr = Ridge(alpha=i)
    model = rr.fit(xtrain,ytrain)
    tr_pred = model.predict(xtrain)
    ts_pred = model.predict(xtest)
    from sklearn.metrics import mean_absolute_error
    tr_err = mean_absolute_error(ytrain,tr_pred)
    ts_err = mean_absolute_error(ytest,ts_pred)
    tr.append(tr_err)
    ts.append(ts_err)


# In[57]:


t = range(0,4000,1)


# In[58]:


import matplotlib.pyplot as plt
plt.plot(t,tr,c="red")
plt.plot(t,ts,c="blue")


# In[59]:


lambdas[-1]


# In[60]:


tr_err


# In[61]:


ts_err


# In[62]:


model


# In[65]:


rr = Ridge(alpha=12.0)
model = rr.fit(xtrain,ytrain)


# In[64]:


xtrain.columns


# # Creating test data ready for predictions

# In[67]:


xtest = test_data.drop(labels=["Id"],axis=1)
xtest_new = preprocessing(xtest)


# In[ ]:


#xtest_new[list(xtrain.columns)]


# In[68]:


xtest_new['Exterior2nd_Other']=0


# In[ ]:


final_data_for_pred = xtest_new[list(xtrain.columns)]


# # predict and save to file

# In[ ]:


pred = model.predict(final_data_for_pred)


# In[ ]:


Q = test[["Id"]]


# In[ ]:


Q['SalePrice']=pred


# In[ ]:


Q.to_csv("C:/Users/Lenovo/Desktop/submissions.csv")

