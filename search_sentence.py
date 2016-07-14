import nltk
import pandas as pd
from edit_distance import stringdist
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import ast
import mysql.connector
from edit_distance import stringdist
import math

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review,"html5lib").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    # print (review)
    raw_sentences = tokenizer.tokenize(review.strip())
    # print (raw_sentences)
    # print (raw_sentences)
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it

        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return raw_sentences,sentences

def find_same_cluster(search_sentence,type_cluster = 3):
    cnx = mysql.connector.connect(user='root', password='',
                                  host='localhost',
                                  database='similarity_search')
    cursor = cnx.cursor()
    # type_cluster = input("select type cluster DBSCAN == 3, KMEAN == 4 : ")
    quert = "Select * from cluster_of_word where book_id = "+str(type_cluster)
    # file_name = 'pg25990.txt'
    # f = open('../Data/'+file_name,'r')

    # book = {'name_of_book':file_name,'text':f.read()}
    cursor.execute(quert)
    # cnx.commit()
    # search_word = input('enter your word : ')
    temp = []
    for i in cursor:
        temp.append(i)
    cnx.close()
    ans_sentence = []
    for search_word in search_sentence:
        ans = []
        min = 1000
        for i in temp:

            if abs(float(i[3])-(1-stringdist(search_word.lower(),str(i[2]).lower(),method = 'cosine',q = 2)[1])) < min:
                min = abs(float(i[3])-(1-stringdist(search_word.lower(),str(i[2]).lower(),method = 'cosine',q = 2)[1]))
                ans = ast.literal_eval(i[5])
                # print (search_word,ans)
            elif abs(float(i[3])-(1-stringdist(search_word.lower(),str(i[2]).lower(),method = 'cosine',q = 2)[1])) == min:
                ans += ast.literal_eval(i[5])
                # print (search_word,ans)

        ans_sentence.append(str(ans))
    return ans_sentence

def intersection(x,y):
    same = []
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] == y[j]:
                same.append(x[i])
                break
    return same
def union(x,y):
    same = []
    for i in range(len(x)):
        if not x[i] in same:
            same.append(x[i])
    for i in range(len(y)):
        if not y[i] in same:
            same.append(y[i])
    return same

def cosine_distance(q1,q2):
    qgramI = len(intersection(q1,q2))*1.0
    # print q1,q2
    # print qgramI
    # print len(q1),len(q2)
    # print math.sqrt(len(q1)*len(q2))
    try:
        distance =  1 - qgramI/math.sqrt(len(q1)*len(q2))
        return distance
    except:
        return 1.0

# x = find_same_cluster([])
# y = find_same_cluster(['mother','others'])
# print (x)
# print (y)
# print (cosine_distance(x,y))
#
def read_from_file(path_of_file):

    f = open(path_of_file,'r')
    data = f.read()
    return data

def get_word_to_list(path_of_file):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    data = read_from_file(path_of_file)
    sentences , list_of_sentences = review_to_sentences(data,tokenizer,remove_stopwords=True)
    return sentences,list_of_sentences

def print_(cosine_dist,topn = 3):
    for i in range(topn):
        print (cosine_dist[i])

def main():

    path_of_file = '../Data/pg25990.txt'
    sentences, list_of_sentences = get_word_to_list(path_of_file)
    x = find_same_cluster(list_of_sentences[0],3)
    print ("SEARCH SENTENCE = "+sentences[0])
    cos_dist = []
    for i in range(len(list_of_sentences)):
        y = find_same_cluster(list_of_sentences[i],3)
        cos_dist.append((sentences[i],1-cosine_distance(x,y)))
    # x = find_same_cluster(,4)
    # print (x)

    # print (y)
    # print (len(intersection(x,y))/len(review_to_sentences(z,tokenizer,remove_stopwords=True)[0]))
    # print (intersection(x,y))

    cos_dist = sorted(cos_dist,key = lambda x: x[1],reverse = True)
    print_(cos_dist)



if __name__ == "__main__":

    main()
