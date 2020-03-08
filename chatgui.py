
# coding: utf-8

# In[13]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model=load_model('chatbot_model.h5')
import json
import random
intents=json.loads(open('intents.json').read())


# In[14]:


words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(wrod.lower()) for word in sentence_words]
    return sentence_words

def box(sentence, words, show_details=True):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1
                if show_details:
                    print("found in bag: %s" %w)
    return(np.array(bag))

def predict_class(sentence,model):
    p=bow(sentence,words,show_details=False)
    res=model.predict(np.array([p]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])})
    return return_list

def getResponse(ints,intents_json):
    tag=ints[0]['intents']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result=random.choice(i['responses'])
            break
    return result
def chatbot_response(text):
    ints=predict_class(text,model)
    res=getResponse(ints,intents)
    return res
                    


# In[15]:


import tkinter
from tkinter import *


# In[16]:


def send():
    msg=EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    
    if msg !='':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END,"You:" + msg + '\n\n')
        ChatLog.config(foreground="#442265",font=("Verdana",12))
    
        res=chatbot_response(msg)
        ChatLog.insert(END,"Bot:" + res + '\n\n')
    
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
    
base=Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE,height=FALSE)
ChatLog=Text(base,bd=0,bg="white",height="8",width="50",font="Arial",)
ChatLog.config(state=DISABLED)
scrollbar=Scrollbar(base,command=ChatLog.yview,cursor="heart")
ChatLog['yscrollcommand']=scrollbar.set


# In[17]:


SendButton=Button(base,font=("Verdana",12,'bold'),text="send",width="12",height=5,bd=0,bg="#32de97",activebackground="#3c9d9b",fg='#ffffff',command=send)


# In[18]:


EntryBox=Text(base,bd=0,bg="white",width="29",height="5",font="Arial")


# In[19]:


scrollbar.place(x=376,y=6,height=386)
ChatLog.place(x=6,y=6,height=386,width=370)
EntryBox.place(x=128,y=401,height=90,width=265)
SendButton.place(x=6,y=401,height=90)
base.mainloop()


# In[11]:





# In[12]:




