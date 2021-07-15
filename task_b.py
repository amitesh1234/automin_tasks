# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:08:10 2021

@author: Zephyrus
"""
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
path="C:/Users/Zephyrus/Desktop/automin/automin-2021-confidential-data/task-B-en/test/"
from sentence_transformers import SentenceTransformer, util

id1=[]
ref=[]
l=[]
count=1

model = SentenceTransformer('bert-base-nli-mean-tokens')
files=os.listdir(path)
for x in files:
    path1=path+x+'/'
    f=os.listdir(path1)
    if len(f)==1:
        continue
    text1=open(path1+f[0],encoding="utf8").read()
    text2=open(path1+f[1],encoding="utf8").read()
    text=[text1,text2]
    documents_df=pd.DataFrame(text,columns=['documents'])
    stop_words_l=stopwords.words('english')
    documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]','',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]','',w).lower() not in stop_words_l) )
      
    # Two lists of sentences
    sentences1 = documents_df.documents_cleaned[0]
      
    sentences2 = documents_df.documents_cleaned[1]
      
    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
      
    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    print(x+' : ', cosine_scores)
    id1.append(count)
    count=count+1
    ref.append(x)
    if cosine_scores.numpy()[0][0]<0.65:
        l.append(False)
    else:
        l.append(True)
final_df=pd.DataFrame()
final_df["Sl.No."]=id1
final_df["Instance Id"]=ref
final_df["Predicted Label"]=l

final_df.to_csv("Task_C_Predications_EN.tsv", sep="\t")
