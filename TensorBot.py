# uses conrnell university movie dialouge dataset


import numpy as np
import tensorflow as tf
import re
import time

# import the dataset.
# split the lines and conversations on new lines '\n'
#preprocessing

lines = open('movie_lines.txt', encoding='utf-8',
             errors='ignore').read().split('\n')

conversations= open('movie_conversations.txt', encoding='utf-8',
                    errors= 'ignore').read().split('\n')


#dictionary for lines mapping
#split per line in +++$+++

lineDict = {} #contains all the lines mapped to key index

for line in lines :
    str = line.split(' +++$+++ ')
    if len(str)== 5:
        lineDict[str[0]]= str[4]


# lis of conversations
#do same as prev step
convList=[]

for conv in conversations[:-1]:  #last row in dataset is empty
    str= conv.split(' +++$+++ ')[-1][1:-1].replace("'", "" ).replace(" ","")
    convList.append(str.split(","))


#Q&A bot : data separation

questions=[]
answers=[]

for conv in convList:
    for i in range(len(conv)-1):
        questions.append(lineDict [conv[i]])
        answers.append(lineDict [conv[i+1]])


#cleaning the text
def clean_text(text):
    text=text.lower()
    text= re.sub(r"i'm", "i am",text )
    text= re.sub(r"he's", "he is",text )
    text= re.sub(r"she's", "she is",text )
    text= re.sub(r"what's", "what is",text )
    text= re.sub(r"where's", "where is",text )
    text= re.sub(r"\'ll", "will",text )
    text= re.sub(r"\'ve", "have",text )
    text= re.sub(r"\'re", "are",text )
    text= re.sub(r"\'d", "would",text)
    text= re.sub(r"won't", "will not", text )
    text= re.sub(r"can't", "cannot not", text )
    text= re.sub(r"wouldn't", "would not", text )
    text= re.sub(r"couldn't", "could not", text )
    text= re.sub(r"[`~$(){}',\"|+-_*!?.:;@#^&]", "", text )
    return text

clean_ques=[]
for q in questions:
    clean_ques.append(clean_text(q))


clean_ans=[]
for a in answers:
    clean_ans.append(clean_text(a))


#word count
wordCount={}

for q in clean_ques:
    for word  in q.split():
        if word not in wordCount:
            wordCount[word]=1
        else:
            wordCount[word] +=1


for a in clean_ans:
    for word  in a.split():
        if word not in wordCount:
            wordCount[word]=1
        else:
            wordCount[word] +=1


#thresholding : a hyperparameter 15-20
threshold=15
ques_words={} # high frequency question words > threshold
ans_words={} #same
wi=0 #word index
#some words have high freqency in question vice versa for answers

for word,count in wordCount.items():
    if count>=threshold:
        ques_words[word]=wi
        wi=wi+1


wi=0

for word,count in wordCount.items():
    if count>=threshold:
        ans_words[word]=wi
        wi=wi+1









