#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression,SGDClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error,classification_report,confusion_matrix
# Building a Regression MLP Using the Sequential API
import tensorflow as tf
from tensorflow import keras


# In[2]:


df=pd.read_csv(r"weatherAUS.csv")
df


# In[3]:


df.info()


# In[4]:


df.describe().T


# In[5]:


print("Shape:",df.shape)
df.head()


# In[6]:


print("Columns:",df.columns,"\n")
df.isnull().mean().sort_values(ascending=False).plot(kind='bar',figsize=(12,4),title='Missing values per each columns') 
#so we will use drop to remove columns -->df.drop([])
plt.ylabel('Missing Ratio')
plt.show()


# In[7]:


print("No.Duplicated=",df.duplicated().sum())
df.drop_duplicates(inplace=True)
df


# In[8]:


df.isnull().sum().sort_values()  #so,we will use dropna to remove rows


# In[9]:


df.dropna(inplace=True)  #Removing rows
df.isnull().sum().sort_values()


# In[10]:


df


# In[11]:


plt.figure(figsize=(14, 10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='Spectral', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[12]:


categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Count Plot for {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[13]:


categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_cols:
    lb = LabelEncoder()
    df[col] = lb.fit_transform(df[col])
    label_encoders[col] = lb


# In[14]:


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace=True)


# In[15]:


df


# In[16]:


x = df.drop("RainTomorrow", axis=1)
y = df['RainTomorrow']


# In[17]:


y.value_counts()


# In[18]:


# Split into Input and Output Elements
from sklearn.model_selection import train_test_split

X_train_full, X_test, y_train_full, y_test = train_test_split(x, 
              y, test_size= 0.20, random_state=42,stratify=y)
# Val set
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full
                                                      , test_size= 0.10,stratify=y_train_full)

print("X_train  = ",X_train.shape ," y_train = ", y_train.shape)
print("X_test   = ",X_test.shape ," y_test = ", y_test.shape)
print("X_valid  = ",X_valid.shape ," y_valid = ", y_valid.shape)


# In[19]:


y[y==0].count()


# In[20]:


scaler = StandardScaler()
X_train_1= scaler.fit_transform(X_train)
X_test_1= scaler.transform(X_test)


# In[21]:


df.head(10)


# In[22]:


'''
lasso = LassoCV()
lasso.fit(X_train_1, y_train)
y_pred_lasso = lasso.predict(X_test_1).round()
print("Lasso Classifier Accuracy:", accuracy_score(y_test, y_pred_lasso))
print("Lasso", y_test, y_pred_lasso)
sns.heatmap(confusion_matrix(y_test, y_pred_lasso), annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Lasso Classifier")
plt.show()
print(classification_report(y_test, y_pred_lasso))
'''


# In[23]:


'''
ridge = RidgeCV()
ridge.fit(X_train_1, y_train)
y_pred_ridge = ridge.predict(X_test_1).round()
print("Ridge Classifier Accuracy:", accuracy_score(y_test, y_pred_ridge))
print("Ridge", y_test, y_pred_ridge)
sns.heatmap(confusion_matrix(y_test, y_pred_ridge), annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Ridge Classifier")
plt.show()
print(classification_report(y_test, y_pred_ridge))
'''


# In[24]:


lr = LogisticRegression()
lr.fit(X_train_1, y_train)
y_pred_lr = lr.predict(X_test_1)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
print(classification_report(y_test, y_pred_lr))


# In[25]:


sgd = SGDClassifier(max_iter=1000)
sgd.fit(X_train_1, y_train)
y_pred_sgd = sgd.predict(X_test_1)
print("SGD Classifier Accuracy:", accuracy_score(y_test, y_pred_sgd))
sns.heatmap(confusion_matrix(y_test, y_pred_sgd), annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - SGD Classifier")
plt.show()
print(classification_report(y_test, y_pred_sgd))


# In[26]:


nb = GaussianNB()
nb.fit(X_train_1, y_train)
y_pred_nb = nb.predict(X_test_1)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()
print(classification_report(y_test, y_pred_nb))


# In[27]:


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_1, y_train)
y_pred_dt = dt.predict(X_test_1)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
print(classification_report(y_test, y_pred_dt))


# In[28]:


rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train_1, y_train)
y_pred_rf = rf.predict(X_test_1)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix - Random Forest")
plt.show()
print(classification_report(y_test, y_pred_rf))


# In[29]:


from tensorflow.keras import layers
model = keras.models.Sequential([
    layers.Dense(8, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')  # For binary classification
])
# Show model summary
print(model.summary())


# In[30]:


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[31]:


# Training and evaluating the model
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, restore_best_weights=True)
history = model.fit(X_train_1, y_train, epochs=50, batch_size=32, validation_data=(X_valid, y_valid),callbacks=[early_stop])


# In[32]:


y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Deep Learning")
plt.show()


# In[33]:


history.history


# In[34]:


# plot the learning curves
pd.DataFrame(history.history).plot(figsize=(12, 8))
plt.grid(True)
#plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

print("-----------------------------------------------------------------------")
# Evaluate the model
model_evaluate = model.evaluate(X_test, y_test)
print("Loss                   : ",model_evaluate[0])
print("Mean Absolute Error     : ",model_evaluate[1])


# In[35]:


from sklearn import metrics

predicted_classes= model.predict(X_test).round()

print("Accuracy : ", metrics.accuracy_score(y_test, predicted_classes))
print("Precision: ", metrics.precision_score(y_test, predicted_classes))
print("Recall   : ", metrics.recall_score(y_test, predicted_classes))
print("F1-score : ", metrics.f1_score(y_test, predicted_classes))


df_data = pd.DataFrame({
    "Actual": np.array(y_test).flatten(),
    "Predicted": predicted_classes.flatten()
})
print("-----------------------------------------------------------------------")
print(df_data.head(15))
print("-----------------------------------------------------------------------")


# In[ ]:





# In[ ]:




