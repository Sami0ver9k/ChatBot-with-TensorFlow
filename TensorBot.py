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











