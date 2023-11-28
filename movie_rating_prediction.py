#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("Movies.csv", encoding='latin1')
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


attributes = df.columns
print(attributes)


# In[6]:


df.isna().sum()


# In[7]:


shape = df.shape
print(f"Number of rows: {shape[0]}, Number of columns: {shape[1]}")


# In[8]:


unique_genres = df['Genre'].unique()
print("Unique Genre:" , unique_genres)


# In[9]:


df.drop_duplicates(inplace=True)


# In[10]:


attribute = ['Name','Year','Duration','Votes','Rating']
df.dropna(subset=attributes,inplace=True)
missing_val = df.isna().sum()
print(missing_val)


# In[11]:


df


# In[12]:


movie_name_rating = df[['Name','Rating']]
print(movie_name_rating.head())


# In[13]:


top_rated_movies = df.sort_values(by = 'Rating', ascending = False).head (10)
plt.figure(figsize = (10, 6))
plt.barh(top_rated_movies['Name'], top_rated_movies['Rating'] ,color = "skyblue")
plt.xlabel ('Rating')
plt.ylabel ('Movie')
# Label for the y-axis (Movie)
plt.title('Top 10 Highest-Rated Movies') 
plt.gca().invert_yaxis()
plt. show()


# In[14]:


df['Votes'] = pd.to_numeric(df['Votes'], errors = "coerce")
plt.figure(figsize = (10, 6))
plt.scatter(df['Rating'], df['Votes'], alpha = 0.5, color= 'b')
plt.xlabel('Rating')
plt.ylabel('Votes')
plt.title('Scatter Plot of Rating vs. Votes')
plt.grid(True)
plt.show()


# In[15]:


actors = pd.concat ([df['Actor 1'], df['Actor 2'], df['Actor 3']])
actor_counts = actors.value_counts().reset_index()
actor_counts.columns = ['Actor','Number of Movies']
plt. figure(figsize = (12, 6))
sns.barplot (x = 'Number of Movies', y = 'Actor', data = actor_counts.head (10), palette = 'viridis')
plt.xlabel('Number of Movies')
plt.ylabel ('Actor')
plt.title('Top 10 Actors by Number of Movies Performed')
plt. show()


# In[16]:


columns_of_interest = ['Votes', 'Rating', 'Duration', 'Year']
sns.set (style ='ticks')
sns.pairplot (df[columns_of_interest], diag_kind ="kde", markers = 'o', palette = 'viridis' , height =2.5,aspect=1.2)
plt.suptitle('Pair Plot of Voting , Rating , Duration, and Year', y = 1.02)
plt. show()


# In[17]:


numerical_columns = ['Votes','Rating', 'Duration', 'Year']
correlation_matrix = df[numerical_columns].corr()
plt. figure (figsize = (8, 5))
sns.heatmap (correlation_matrix, annot = True, cmap = 'coolwarm', vmin = -1, vmax = 1)
plt.title('Correlation Heatmap')
plt.show()


# In[18]:


df_sorted = df.sort_values(by = 'Votes', ascending = False)
#df_sorted['Vote_Count_Percentile'] = df_sorted['Votes'].rank(pct = True) * 100
df_sorted.reset_index(drop = True,inplace = True)
print (df_sorted[['Name','Votes', ]])


# In[19]:


df.head()


# In[20]:


df = df.dropna(subset = ['Votes'])
df.head()


# In[21]:


df[ 'Year'] = df['Year'].astype(str)
df[ 'Duration'] = df['Duration'].astype(str)
df['Year'] = df['Year'].str.extract('(\d+)').astype(float)
df['Duration'] = df[ 'Duration'].str.extract('(\d+) ').astype(float)
X=df[['Year','Duration','Votes']]
y=df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[22]:


model = LinearRegression()


# In[23]:


model.fit(X_train , y_train)


# In[24]:


y_pred = model.predict(X_test)


# In[25]:


mae = mean_absolute_error (y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared = False)
r2 = r2_score(y_test, y_pred)
print (f"Mean Absolute Error: {mae}")
print (f"Root Mean Squared Error: {rmse}")
print (f"R-squared (R2) Score: {r2}")


# In[26]:


#Pair Plots and Correlation Matrix
import seaborn as sns

sns.pairplot(df)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr(method='spearman')


# In[27]:


#model deployment
non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
print(non_numeric_columns)


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

# Generate random data for demonstration
y_test = np.random.rand(100) * 10  # Actual ratings
y_pred = np.random.rand(100) * 10  # Predicted ratings
errors = y_test - y_pred

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Scatter plot
axs[0].scatter(y_test, y_pred)
axs[0].set_xlabel("Actual Ratings")
axs[0].set_ylabel("Predicted Ratings")
axs[0].set_title("Actual vs. Predicted Ratings")

# Line plot
movie_samples = np.arange(1, len(y_pred) + 1)
axs[1].plot(movie_samples, y_pred, marker='o', linestyle='-')  # Use a valid linestyle
axs[1].set_xlabel("Movie Samples")
axs[1].set_ylabel("Predicted Ratings")
axs[1].set_title("Predicted Ratings Across Movie Samples")
axs[1].tick_params(axis='x', rotation=45)

# Histogram
axs[2].hist(errors, bins=30)
axs[2].set_xlabel("Prediction Errors")
axs[2].set_ylabel("Frequency")
axs[2].set_title("Distribution of Prediction Errors")
axs[2].axvline(x=0, color='r', linestyle='--')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

