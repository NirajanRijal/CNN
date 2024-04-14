#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
import pickle


# Lets import image

# In[2]:


DIRECTORY=r'C:\Users\USER\Downloads\valid'
CATEGORIES = ['cats', 'dogs']


# Reading image and using cv2 converting it to array

# In[3]:


data = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = CATEGORIES.index(category)
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_arr = cv2.resize(arr, (100, 100))  # image size may impact computer speed
        data.append([new_arr, label])


# In[4]:


data


# So what we did in above code:-
# 
# 1. It starts with an empty list called data.
# 2. It goes through each category in a list called CATEGORIES - cats & dogs.
# 3. For each category, it creates a path by joining a directory path (DIRECTORY) with the category name.
# 4. Inside each category's path, it looks at each image file.
# 5. For each image, it creates a full image path by joining the category path with the image file name.
# 6. It assigns a label to the image based on its category.
# 7. It reads the image using a library called OpenCV, converting it to grayscale.
# 8. It resizes the image to be 60x60 pixels.
# 9. It adds both the resized image and its label to the data list.

# In[5]:


random.shuffle(data)   # If the data is not shuffled, the model might inadvertently learn patterns based on the order of the data. For example,
#if all the cat images are seen before any dog images, the model might start leaning more towards 
#classifying everything as a cat because it's seen more cat examples recently.


# In[6]:


X = []
y = []


# In[7]:


for features, label in data:
    X.append(features)
    y.append(label)


# In[8]:


X = np.array(X)
y = np.array(y)


# In[9]:


X


# In[10]:


y


# In[11]:


pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(y, open('y.pkl', 'wb'))

