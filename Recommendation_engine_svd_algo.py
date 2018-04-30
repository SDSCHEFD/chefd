
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from datetime import datetime
import random


# In[9]:


customer_recipes_purchase = pd.read_csv('customer_recipes_purchase_count_new.csv',usecols = ['user_id_new','recipe_name','counts'])
customer_recipes_purchase.head()


# In[10]:


#customer_recipes_purchase.loc[customer_recipes_purchase.index[-1]+1] = {'user_id_new': 19596, 'recipe_name': "Fuji Apple Coleslaw", 'counts': 1}
customer_recipes_purchase.head(15)


# In[11]:


customer_recipes = pd.read_csv('customer_recipes_purchase_count.csv',usecols = ['recipe_name'])
#customer_recipes.loc[customer_recipes.index[-1]+1] = {'recipe_name': 'Fuji Apple Coleslaw'}
customer_recipes.tail()


# In[12]:


default_start_time = datetime.now()
start_time = datetime.now()
customer_recipes_pivot = customer_recipes_purchase.pivot(index='user_id_new', columns='recipe_name', values='counts').fillna(0)
customer_recipes_pivot.tail()
print(
            "Time taken to convert to pivot table: %s" % str(
                    datetime.now() - start_time
                )[:-3]
            )


# In[13]:


customer_recipes_pivot.head()


# In[14]:


default_start_time = datetime.now()
start_time = datetime.now()

c_p = customer_recipes_pivot.as_matrix()
user_purchase_mean = np.mean(c_p, axis = 1)
c_p_demeaned = c_p - user_purchase_mean.reshape(-1, 1)

print(
            "Time taken to convert to Matrix: %s" % str(
                    datetime.now() - start_time
                )[:-3]
            )


# In[15]:


default_start_time = datetime.now()
start_time = datetime.now()

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(c_p_demeaned, k = 50)

print(
            "Time taken to convert to Train: %s" % str(
                    datetime.now() - start_time
                )[:-3]
            )


# In[16]:


sigma = np.diag(sigma)


# In[17]:


all_user_predicted_purchase = np.dot(np.dot(U, sigma), Vt) + user_purchase_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_purchase, columns = customer_recipes_pivot.columns)
preds_df.head()


# In[18]:


def recommend_recipes(predictions_df, userID, customer_recipes, customer_recipes_purchase, num_recommendations=5):
    
    
    user_row_number = userID - 1
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    
    user_data = customer_recipes_purchase[customer_recipes_purchase.user_id_new == (userID)]
    user_full = (user_data.merge(customer_recipes, how = 'left', left_on = 'recipe_name', right_on = 'recipe_name')
                 )
    user_full = user_full.drop_duplicates()
    recommendations = (customer_recipes[~customer_recipes['recipe_name'].isin(user_full['recipe_name'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'recipe_name',
               right_on = 'recipe_name').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False)
                      )
    recommendations = recommendations.drop_duplicates()

    return user_full, recommendations.iloc[:num_recommendations, :-1]


# In[28]:


already_purchased, predictions = recommend_recipes(preds_df, 25,customer_recipes, customer_recipes_purchase, 5)


# In[29]:


already_purchased.head()


# In[30]:


predictions

