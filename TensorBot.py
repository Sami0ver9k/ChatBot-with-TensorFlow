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



tokens=['<PAD>', '<EOS>', '<OUT>', '<SOS>']


for token in tokens:
    ques_words[token]=len(ques_words)+1


for token in tokens:
    ans_words[token]=len(ans_words)+1



#swap dic akd= pppppppp'

swapped_ans_words={indx:word  for  word,indx in ans_words.items() }

#adding eos to ans
for i in range(len(clean_ans)):
    clean_ans[i] +=  ' <EOS>'

#converting words into int for easy sorting

ques_into_int=[]

for q in   clean_ques:
    intVals=[]
    for word in q.split():
        if word not in ques_words:
            intVals.append(ques_words['<OUT>'])

        else:
            intVals.append(ques_words[word])

    ques_into_int.append(intVals)



ans_into_int=[]
for a in   clean_ans:
    intVals=[]
    for word in a.split():
        if word not in ans_words:
            intVals.append(ques_words['<OUT>'])

        else:
            intVals.append(ques_words[word])

    ans_into_int.append(intVals)


#sorting
clean_sorted_ques=[]
sorted_index=[]
clean_sorted_ans=[]

list_of_sizes=[]

for i in range (len(ques_into_int)):
    list_of_sizes.append(len(ques_into_int[i]))


max_ques_length=max(list_of_sizes)


for length in range (1,30+1):
    for obj in enumerate(ques_into_int):
        if len(obj[1])==length:
            clean_sorted_ques.append(ques_into_int[obj[0]])
            clean_sorted_ans.append(ans_into_int[obj[0]])


#end of text processing part


###### tensorflow input

def model_input():
    train_input= tf.placeholder(tf.int32, [None,None], name='train_input')
    target_output=tf.placeholder(tf.int32,[None,None], name='target_output')
    learning_rate=tf.placeholder(tf.float32,name='learning_rate')
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    return train_input, target_output, learning_rate, keep_prob











