
# coding: utf-8

# In[1]:


import nltk


# In[2]:


nltk.download('punkt')


# In[3]:


nltk.download('wordnet')


# In[4]:


from nltk.stem import WordNetLemmatizer


# In[5]:


lemmatizer=WordNetLemmatizer()


# In[6]:


import json


# In[7]:


import pickle


# In[8]:


import numpy as np


# In[9]:


from keras.models import Sequential


# In[10]:


from keras.layers import Dense, Activation, Dropout


# In[11]:


from keras.optimizers import SGD


# In[12]:


import random


# In[13]:


words=[]


# In[14]:


classes=[]


# In[15]:


documents=[]


# In[16]:


ignore_words=['?','!']


# In[17]:


data_file=open('intents.json').read()


# In[18]:


intents=json.loads(data_file)


# In[19]:


for intent in intents['intents']:
    for pattern in intent['patterns']:  
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[20]:


words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))


# In[21]:


classes=sorted(list(set(classes)))


# In[22]:


print(len(documents),"documents")


# In[23]:


print(len(classes),"classes",classes)


# In[24]:


print(len(words),"unique lemmatized words",words)


# In[25]:


pickle.dump(words,open('words.pkl','wb'))


# In[26]:


pickle.dump(classes,open('classes.pkl','wb'))


# In[27]:


training=[]


# In[28]:


output_empty=[0]*len(classes)


# In[29]:


for doc in documents:
    bag=[]
    pattern_words=doc[0]
    pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]
for w in  words:  
    bag.append(1) if w in pattern_words else bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    training.append([bag,output_row])
    

        


# In[30]:


random.shuffle(training)


# In[31]:


training=np.array(training)


# In[32]:


train_x=list(training[:,0])
train_y=list(training[:,1])
print("Training data created")


# In[33]:


model=Sequential()


# In[34]:


model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))


# In[35]:


model.add(Dropout(0.5))


# In[36]:


model.add(Dense(64,activation='relu'))


# In[37]:


model.add(Dropout(0.5))


# In[38]:


model.add(Dense(len(train_y[0]),activation='softmax'))


# In[39]:


sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)


# In[40]:


model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])


# In[41]:


hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)


# In[42]:


model.save('chatbot_model.h5',hist)


# In[43]:


print("model created")

