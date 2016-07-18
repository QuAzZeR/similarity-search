import pandas as pd
import ast
import math
import multiprocessing as mp
import os
import time

from clustering_lib import find_cluster
from edit_distance import stringdist
from sentence_lib import get_word_from_document
word_distance1 = mp.Queue()
word_distance2 = mp.Queue()




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

# x = find_cluster([])
# y = find_cluster(['mother','others'])
# print (x)
# print (y)
# print (cosine_distance(x,y))
#
def read_from_file(path_of_file):

    f = open(path_of_file,'r')
    data = f.read()
    return data



def print_(cosine_dist,topn = 3):
    for i in range(topn):
        print (cosine_dist[i])

def read_from_file(path_of_file):
    tata = open(path_of_file,'r+')
    text = tata.readlines()
    list_of_sentences_encoding = []
    for i in text:
        list_of_sentences_encoding.append(ast.literal_eval(i))
    return list_of_sentences_encoding

def cal_distant_parallel(args):

    parallel_dist = 1-cosine_distance(args[1],args[1][3])
    # if args[0] == 1:
    word_distance1.put((args[1][2],parallel_dist))
    return (args[1][2],parallel_dist)

    # else:


def main():
    # print(os.getpid())
    # path_of_file = '../Data/pg25990.txt'
    list_of_sentences_encoding = read_from_file('./test_prepare5000')
    # print (len(list_of_sentences_encoding))
    pool = mp.Pool(mp.cpu_count())
    number_of_sentence = len(list_of_sentences_encoding)


    search_sentence = input()
    sentence, sentence_with_word = get_word_from_document(search_sentence)
    # print(sentence,sentence_with_word)
    search_sentence_with_encoding = find_cluster(sentence_with_word[0],4)
    list_for_parallel = [(search_sentence_with_encoding,i) for i in list_of_sentences_encoding]

    # print (type())
    # list_for_parallel2 = [(2,search_sentence_with_encoding,i) for i in list_of_sentences_encoding[len_1:]]
    # pool.map(cal_distant_parallel,list_for_parallel2)
    start =  time.time()
    # pool.map(cal_distant_parallel,list_for_parallel)
    # list_of_distance = [word_distance1.get() for i in range(number_of_sentence)]
    list_of_distance = [cal_distant_parallel(i) for i in list_for_parallel]
    end = time.time()
    list_of_distance = sorted(list_of_distance,key = lambda x: x[0])

    #     list_of_distance.append(x)
    print(end-start)
    print(len(list_of_distance))
    # for i in list_of_distance:
    #     print (i)

    # sentences, list_of_sentences = get_word_from_document(path_of_file)
    # x = find_cluster(list_of_sentences[1],4)
    # print(len(sentences))
    # print ("SEARCH SENTENCE = "+sentences[1])
    # cos_dist = []
    # for i in range(len(list_of_sentences)):
    #     y = find_cluster(list_of_sentences[i],4)
    #     # cos_dist.append((sentences[i],1-cosine_distance(x,y)))
    # x = find_cluster(,4)
    # print (x)

    # print (y)
    # print (len(intersection(x,y))/len(review_to_sentences(z,tokenizer,remove_stopwords=True)[0]))
    # print (intersection(x,y))

    # cos_dist = sorted(cos_dist,key = lambda x: x[1],reverse = True)
    # print_(cos_dist)

if __name__ == "__main__":

    main()
