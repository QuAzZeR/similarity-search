import pandas as pd
import ast
import math
import multiprocessing as mp
import os
import time
import sqlite3

from clustering_lib import find_cluster
from edit_distance import stringdist
from sentence_lib import get_word_from_document
word_distance1 = mp.Queue()
word_distance2 = mp.Queue()

l1 = mp.RLock()
l2 = mp.RLock()


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
    # try:
    # print(len(q1),len(q2))
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
# def read_from_file(path_of_file):
#
#     f = open(path_of_file,'r')
#     data = f.read()
#     return data



def print_(cosine_dist,topn = 1):
    for i in range(topn):
        print (cosine_dist[i])

def read_from_file(path_of_file):#,search_sentence_encoding):
    tata = open(path_of_file,'r+')
    text = tata.readlines()
    tata.close()
    return text
    # list_of_sentences_encoding = []
    # list_of_distance = []
    #
    # for i in text:
    #     # list_of_sentences_encoding.append(ast.literal_eval(i))
    #     x = ast.literal_eval(i)
    #     list_of_distance.append(cal_distant_parallel((search_sentence_encoding,x)))
    # return list_of_distance

def cal_distant_parallel(args):#(search_sentence_encoding,list_of_sentences_encoding,ans):
    # print (args)
    global l
    # for i in range(len(list_of_sentences_encoding)):
    #     # print(list_of_sentences_encoding[i][3])
    #     # print list_of_sentences_encoding
    #     print (ans[i])
    #     ans[i] = (list_of_sentences_encoding[i][2],1-cosine_distance(search_sentence_encoding,list_of_sentences_encoding[i][3]))
    #     print (ans[i])
    # print (args)
    x = ast.literal_eval(args[1])
    # print (x[3])
    # # while not l.acquire(timeout = 0.01):
    #     pass
    parallel_dist = 1-cosine_distance(args[0],x[3])
    print (parallel_dist)
    # parallel_dist = 1-cosine_distance(args[0],args[1][3])
    # print (parallel_dist)
    # # if args[0] == 1:
    # word_distance1.put((args[1][2],parallel_dist))
    if args[2]%2 == 0:
        l1.acquire()
        word_distance1.put((x[2],parallel_dist))
        l1.release()
    else:
        l2.acquire()
        word_distance2.put((x[2],parallel_dist))
        l2.release()
    # return (args[1][2],parallel_dist)

    # else:
def cal_distant_sequential(args):#(search_sentence_encoding,list_of_sentences_encoding,ans):
    # print (args)
    # for i in range(len(list_of_sentences_encoding)):
    #     # print(list_of_sentences_encoding[i][3])
    #     # print list_of_sentences_encoding
    #     print (ans[i])
    #     ans[i] = (list_of_sentences_encoding[i][2],1-cosine_distance(search_sentence_encoding,list_of_sentences_encoding[i][3]))
    #     print (ans[i])
    # print (args)
    x = ast.literal_eval(args[1])
    parallel_dist = 1-cosine_distance(args[0],x[3])
    # print(args)
    # x =args[1]
    # print (x[3])
    # parallel_dist = 1-cosine_distance(args[0],ast.literal_eval(x[3]))
    # print (parallel_dist)
    # parallel_dist = 1-cosine_distance(args[0],args[1][3])
    # print (parallel_dist)
    # # if args[0] == 1:
    # word_distance1.put((args[1][2],parallel_dist))
    # word_distance1.put((x[2],parallel_dist))
    return (x[2],parallel_dist)

    # else:


def main():
    # print(os.getpid())
    # path_of_file = '../Data/pg25990.txt'

    # print (len(list_of_sentences_encoding))
    start_all  =time.time()
    # pool = mp.Pool(mp.cpu_count())




    search_sentence = input()
    sentence, sentence_with_word = get_word_from_document(search_sentence)
    # print(sentence,sentence_with_word)
    # time.sleep(1.0)
    search_sentence_with_encoding = find_cluster(sentence_with_word[0],4)
    list_of_sentences_encoding = read_from_file('./test_prepare.txt')
    print("FINISHED READ FILE")
    # list_of_distance = read_from_file('./test_prepare.txt',search_sentence_with_encoding)
    # number_of_sentence = len(list_of_sentences_encoding)
    # print (list_of_sentences_encoding[0])

    list_for_parallel = [(search_sentence_with_encoding,list_of_sentences_encoding[i],i) for i in range(len(list_of_sentences_encoding))]
    print("FINISHED PREPARE FILE FOR CALCULATE DISTANCE")
    # SIZE = len(list_for_parallel)

    # print (list_for_parallel[2][1])
    # return 0

    start =  time.time()
    # print("PARALLEL")
    # pool = mp.Pool(mp.cpu_count())
    # pool.map(cal_distant_parallel,list_for_parallel)
    # list_of_distance = [word_distance1.get() for i in range(int(SIZE/2))]
    # list_of_distance = [word_distance2.get() for i in range(int(SIZE/2))]
    # N = 4
    # p = []
    #
    # for i in range(1,N):
    #     p.append(mp.Process(target = cal_distant_parallel,args =(search_sentence,list_of_sentences_encoding[(i-1)*int(number_of_sentence/N):(i)*int(number_of_sentence/N)],list_of_distance[(i-1)*int(number_of_sentence/N):(i)*int(number_of_sentence/N)],)))
    # p.append(mp.Process(target = cal_distant_parallel,args =(search_sentence,list_of_sentences_encoding[(i-1)*int(number_of_sentence/N):],list_of_distance[(i-1)*int(number_of_sentence/N):],)))
    # print ((len(p)))
    # for i in p:
    #     i.start()
    #     i.join()
    # print


    # for i in c.execute("SELECT * FROM sentence_encoding"):
    print("SEQUENTIAL LOAD ALL SENTENCE TO MEMORY")
    # print("SEQUENTIAL DIDN'T LOAD ALL SENTENCE TO MEMORY")
    # conn  = sqlite3.connect('sentence_encoding.db')
    # c = conn.cursor()
    list_of_distance = [cal_distant_sequential(i) for i in list_for_parallel]
    # list_of_distance = [cal_distant_sequential((search_sentence_with_encoding,i)) for i in c.execute("SELECT * FROM sentence_encoding")]

    # conn.close()
    end = time.time()
    print("FINISHED RUN")
    SIZE = len(list_of_distance)
    print ("SIZE  = " +str(SIZE))
    # print(list_of_distance[0])
    print("time for calculate similarity distance = "+str(end-start))
    print(str(SIZE/(end-start))+" Sentences/Second")

    list_of_distance = sorted(list_of_distance,key = lambda x: -x[1])

    # def cal_distance_with_sqlite(search_sentence_encoding):


    # print(list_of_distance[0])
    #     list_of_distance.append(x)


    # print_(list_of_distance,3)
    end_all = time.time()
    print("All Process Time= "+str(end_all-start_all))
    print(str(SIZE/(end_all-start_all))+" Sentences/Second")
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
