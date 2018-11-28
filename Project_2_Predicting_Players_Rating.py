
# coding: utf-8

# <h3>Predicting players rating</h3>
# <p>In this project I am going to predict the overall rating of soccer player based on their attributes such as 'crossing', 'finishing etc.
# <p>The dataset I am going to use is from European Soccer Database (https://www.kaggle.com/hugomathien/soccer) has more than 25,000 matches and more than 10,000 players for European professional soccer seasons from 2008 to 2016.
# 
# <h3>About the Dataset</h3>
# <p>The ultimate Soccer database for data analysis and machine learning
# The dataset comes in the form of an SQL database and contains statistics of about 25,000 football matches, from the top football league of 11 European Countries. It covers seasons from 2008 to 2016 and contains match statistics (i.e: scores, corners, fouls etc...) as well as the team formations, with player names and a pair of coordinates to indicate their position on the pitch.
# 
# <li>+25,000 matches
# <li>+10,000 players
# <li>11 European Countries with their lead championship
# <li>Seasons 2008 to 2016
# <li>Players and Teams' attributes* sourced from EA Sports' FIFA video game series, including the weekly updates
# <li>Team line up with squad formation (X, Y coordinates)
# <li>Betting odds from up to 10 providers
# 
# <p>Detailed match events (goal types, possession, corner, cross, fouls, cards etc...) for +10,000 matches
# The dataset also has a set of about 35 statistics for each player, derived from EA Sports' FIFA video games. It is not just the stats that come with a new version of the game but also the weekly updates. So for instance if a player has performed poorly over a period of time and his stats get impacted in FIFA, you would normally see the same in the dataset.
#     
# <h3>Importing Required Modules</h3>

# In[1]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import statsmodels.api as statsmodels

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, make_scorer

import math
from math import sqrt


# <h3>Data Pre-Processing</h3>

# In[2]:


#Creates a connection with database.sqlite db
conn = sqlite3.connect('database.sqlite')

#Reads sql statement from Player_Attributes from Pandas and put into dataframe for further data exploration and processing
df = pd.read_sql_query('select * from Player_Attributes',conn)


# <h3>Data Exploration</h3>

# In[3]:


#df.head(10) return first 10 rows of the data frame and transpose converts rows to column
#This gives very good view of data through which we can observe columns/features, their values and data types, etc.
df.head(10).transpose()


# In[4]:


#Second thing we need to know about columns or features of a particular database which can be achieved by df.columns
df.columns 


# In[5]:


#Statistical summary of the dataframe df or data
#Count     : Number of rows per column or feature contains.
#mean      : Average value for a particular column.
#std       : Standard deviation is SqaureRoot(Summation(x-x_bar))
#50%       : The median and it's difference from the mean gives information on the skew of the distribution. 
#            It's also another definition of average that is robust to outliers in the data.
#25% & 75% : Perspective on the kurtosis. All percentile numbers are generally more robust to outliers.
df.describe().transpose()


# In[6]:


#Return a tuple representing the dimensionality of the DataFrame df.
#In this tuple, first entity represents number of rows and second entity represents number of columns
df.shape 


# <h3>Data Visualization</h3>

# <h5>Histogram Plot</h5>

# In[7]:


# Visual Analysis of the Dataset
df.hist(bins=30, figsize=(30,30))
plt.show()


# In[8]:


#Corelations 
df.corr()


# In[9]:


# Show the Coorelations in Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr())
plt.show()


# <b>Observation 1:</b> All variables are positively corelated with Overall Rating except id, player_fifa_api_id, player_api_id. So these variables can be removed from the Regression Model.

# In[36]:


# Checking impact of Categorical Columns

# Impact of Preferred Foot.
sns.barplot(x='preferred_foot', y='overall_rating', data=df, estimator=np.mean)
plt.title('Preferred Foot Vs Overall Rating')
plt.show()

# Impact of Attacking Work Rate.
sns.barplot(x='attacking_work_rate', y='overall_rating', data=df, estimator=np.mean)
plt.title('Attacking Work Rate Vs Overall Rating')
plt.show()

# Impact of Defensive Work Rate.
sns.barplot(x='defensive_work_rate', y='overall_rating', data=df, estimator=np.mean)
plt.title('Defensive Work Rate Vs Overall Rating')
plt.show()


# <b>Observation 2:</b> Here the Overall Rating is almost same for the different values of preferred_foot, attacking_work_rate, defensive_work_rate. So these variables can be removed from the Regression Model

# <h3>Data Cleaning</h3>

# In[11]:


#Check for any NA’s in the dataframe.
df.isnull().values.any()
# Null check
df.isnull().sum(axis=0)


# In[12]:


#Drop the rows where at least one element is missing.
df1 = df.dropna() 


# In[13]:


#Return a tuple representing the dimensionality of the DataFrame df1.
df1.shape 


# In[14]:


#Columns of dataframe df1
df1.columns 


# In[15]:


# Null check
df1.isnull().sum(axis=0)


# In[16]:


#Return a tuple representing the dimensionality of the DataFrame df1.
df1.shape 


# In[17]:


# Identifying Indipendant variables for the Model
target = ['overall_rating']            # Target variable

# To Identify the feature first we will take all column in the list then remove 
# which is not relivant (as per Observation-1 and Observation-2)
features = list(df1)

features.remove('overall_rating')      # Removed as - Target Variable
features.remove('id')                  # Removed as - Observation 1
features.remove('player_fifa_api_id')  # Removed as - Observation 1
features.remove('player_api_id')       # Removed as - Observation 1
features.remove('date')                # Removed as - Observation 1

features.remove('preferred_foot')      # Removed as - Observation 2
features.remove('attacking_work_rate') # Removed as - Observation 2
features.remove('defensive_work_rate') # Removed as - Observation 2

# now the list features has Columns to be considered for Indipendant variables.
features


# In[18]:


# Create Data Frame for both Target and Independant Variables.
X=df1[features]
Y=df1[target]

# Create a Model and Analyze stats
model = statsmodels.OLS(Y, X).fit()
model.summary()


# <b>Observation 3:</b>
# 
# R-squared = 0.998 i.e. 99.80% . So the Model can be considered as Perfect
# For only Balance and Vision, P Values is greater than 0.05. So this two Variables are impacting on Overall Rating
# 
# <b>Train and Test Split</b>
# Train Set 70%, Test Set 30%

# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
print('X_train:\t',X_train.shape)
print('Y_train:\t',Y_train.shape)
print('X_test:\t\t',X_test.shape)
print('Y_test:\t\t',Y_test.shape)


# In[20]:


# Linear Regression 
model_lr = LinearRegression()
model_lr.fit(X_train, Y_train)
pd.DataFrame({'Colum_name':X_train.columns,'Coefcient':model_lr.coef_[0]}) # show Coefcient


# <h4>Evaluate Model Performance</h4>
# <h5>Score</h5>

# In[21]:


print('Train:\t',model_lr.score(X_train,Y_train))
print('Test:\t',model_lr.score(X_test,Y_test))


# <b>Conclusion:</b> Model Score is almost same for Test and Train. i.e. the fit is good.
# 
# <h4>Overall rating Prediction</h4>

# In[22]:


# Calcualte Prediction for Train and Test Data set
y_prediction_train= model_lr.predict(X_train)
y_prediction_test = model_lr.predict(X_test)

print('Train:\t',y_prediction_train.mean())
print('Test:\t', y_prediction_test.mean())


# <b>Conclusion:</b> Overall rating Prediction is almost same for Test and Train. i.e. the fit is good.
# 
# <h3>Prediction for RMSE</h3>

# In[23]:


print('Train:\t', math.sqrt(mean_squared_error(y_prediction_train, Y_train)))
print('Test:\t' , math.sqrt(mean_squared_error(y_prediction_test, Y_test)))


# Plot using Train Data with Residual

# In[24]:


#plt.scatter(model_reg.predict(X_train),model_reg.predict(X_train)-Y_train,c='b',s=40,alpha=0.5)
plt.scatter(y_prediction_train,(y_prediction_train-Y_train),c='b',s=40,alpha=0.5)
plt.hlines(y=0,xmin=0,xmax=100)
plt.title('Residual plot using training Data')
plt.ylabel('Residual')
plt.show()

#Plot using Test Data with caluclated Residual
plt.scatter(y_prediction_test,(y_prediction_test-Y_test),c='y',s=40)
plt.hlines(y=0,xmin=0,xmax=100)
plt.title('Residual plot using test data')
plt.ylabel('Residual')


# <b>Conclusion:</b>
# The residual are randomly scattered around line zero (for both Train and Test Dataset), so the Model is Perfect and Ready to use.
# 
# Now we will Predict Rating for a player to know overall rating by passing the features.

# In[25]:


# For the inut feature we are getting feature of an random player and passing the feature 
# to get the overall rating (later we can compare with his rating as we know his rating here)
input_feature = features[:]
pred_overall_rating=model_lr.predict(np.array(df1[input_feature][999:1000]))
print('Predicted Overall Rating:\t', pred_overall_rating)
print('Actual Overall Rating:\t\t', df1['overall_rating'][999:1000].values[0])


# In[26]:


Y_pred = model_lr.predict(X_test) #Calculating the prediction values


# In[27]:


Y_pred.shape #Prediction shape from test data


# <h3>Predicting overall_rating using Test Data</h3>

# In[28]:


#To visualize the differences between actual overall rating and predicted values, creating a scatter plot.
sns.set_style("whitegrid")
sns.set_context("poster")
plt.figure(figsize=(16,9))
plt.scatter(Y_test, Y_pred)
plt.xlabel("Overall Rating: $Y_i$")
plt.ylabel("Predicted Overall Rating: $\hat{Y}_i$")
plt.title("Overall Rating vs Predicted Overall Rating: $Y_i$ vs $\hat{Y}_i$")
plt.text(40,25, "Comparison between the actual Overall Rating and predicted Overall Rating.", ha='left')
plt.show()


# In[29]:


sns.regplot(x=model_lr.predict(X), y=df1['overall_rating'], fit_reg=True) #Plot Y_test and Y_pred for Linear Regression Model.


# Model Evaluation Using Cross-Validation

# In[30]:


#Evaluating the model using 10-fold cross-validation
scores = cross_val_score(LinearRegression(), X, Y, scoring='neg_mean_squared_error', cv=10)
scores


# In[31]:


np.sqrt(scores.mean() * -1)


# In[32]:


print("The Root Mean Square Error using cross validation for the Model is "+ str(np.sqrt(scores.mean() * -1)) +" and the Results can be further improved using feature extraction and rebuilding, training the model.")


# Evaluating the Model Using RMSE

# In[33]:


#Calculating Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred) #Mean Squared Error: To check the level of error of a model
print(mse)


# In[34]:


#Calculating Root Mean Squared Error#Calcula 
rmse = mse ** 0.5 #Square root of mse (Mean Squared Error)
print(rmse)


# In[35]:


print("The Root Mean Square Error (RMSE) for the Model is "+ str(rmse) +" and the Results can be further improved using feature extraction and rebuilding, training the model.")

