#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


X=pickle.load(open('X.pkl','rb'))
y=pickle.load(open('y.pkl','rb'))


# In[3]:


X=X/255


# In[4]:


import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
import numpy as np


# In[5]:


# Assuming X is your input data (a 3D tensor)
X_expanded = np.expand_dims(X, axis=-1)

model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', padding='valid',input_shape=X_expanded.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),padding='valid'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),padding='valid'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))


# In[6]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[8]:


history=model.fit(X, y, epochs=6, validation_split=0.1)


# In[9]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], color='red',label='train')
plt.plot(history.history['val_accuracy'], color='yellow',label='validation')
plt.legend()


# In[ ]:




