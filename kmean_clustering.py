from __future__ import print_function
# from sklearn.datasets import make_blobs
# from sklearn.cluster import SpectralClustering
# from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from edit_distance import stringdist
import math
import nltk
# import string
import numpy as np
import multiprocessing as mp
import os
from sentence_lib import remove_stopword
import time
import csv
import sys
import sqlite3



queue_list_word = mp.Queue()
distance_queue = mp.Queue()
cluster_queue = mp.Queue()
lock = mp.Lock()
lock2 = mp.Lock()
list_of_word_global = []

def uniq_word(list_non_uniq):
    return list(set(list_non_uniq))

def read_from_file(args):
    document = args
    # file_name = args
    # print(file_name)
    # print
    # open_file = open('../Data/'+file_name,'r')
    # list_of_words = get_word_from_text(open_file)
    # open_file.close()


    list_of_words = get_word_from_text(document)
    # print(len(list_of_words))
    lock.acquire()
    queue_list_word.put(list_of_words)
    lock.release()

def read_from_file_csv(file_name):
    # file_name =
    with open('../Data/'+file_name,'r',encoding='utf-8', errors='ignore') as csvfile:
        a = csv.reader(csvfile)
        print (a)
        x= []
        for i in a:
            # print(i[1])
            # temp = unicode(i[1], errors='replace')
            x.append(i[1])
    print(sys.getsizeof(x))
    return x
# def get_word_from_text(open_file):
def get_word_from_text(document):
    # list_of_words = nltk.word_tokenize(open_file.read())
    list_of_words = nltk.word_tokenize(document)
    list_of_words = uniq_word(list_of_words)
    list_of_words = remove_stopword(list_of_words)
    return list_of_words

def calculate_distance(args):
    number = args[0]
    compare_word = args[1]
    list_of_words = args[2]

    z =[]
    for word in list_of_words:
        w = stringdist(compare_word ,word,q = 2)
        z.append(1-w[1])
    lock2.acquire()
    distance_queue.put((number,compare_word,z))
    lock2.release()

def kmean_clustering(args):

    compare_word = args[0]
    word_distance = [(0,args[1][i]) for i in range(len(args[1]))]
    # print(word_distance)
    # print(word_distance)
    # print(compare_word+' word_distance  = '+str(len(word_distance)))
    # word_distance = []
    # for word in list_of_word_global:
    #     w = stringdist('limited' ,word,q = 2)
    #     word_distance.append((0,w[1]))
    X = np.asarray(word_distance)


    range_n_clusters = range(3,100)
    Max = 0
    index = 0
    Z = 0
    count = 0
    BEFORE = 0


    for n_clusters in range_n_clusters:
        print(compare_word+' cluster = '+str(n_clusters))

        clusterer = KMeans(n_clusters=n_clusters ,init='k-means++',n_jobs=1)
        # print(1)
        cluster_labels = clusterer.fit_predict(X)
        # print(2)
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print(3)
        # print (Max,silhouette_avg,count)
        if Max < silhouette_avg:
            Max = silhouette_avg
            index = n_clusters
            Z = cluster_labels
            # print (index,count)


        if BEFORE == silhouette_avg:
            count+=1
        else :
            count = 1
            BEFORE = silhouette_avg
        if count == 5 or Max == 1:
            break
        # print(4)
        print("For n_clusters = "+str(n_clusters)+
              " The average silhouette_score is : "+str(silhouette_avg))
        # print(5)
        # count+=1
    # fi.write(str(index)+"\n")
    q = [(compare_word,list_of_word_global[i],word_distance[i],Z[i]) for i in range(len(Z))]
    q = sorted(q,key=lambda x: -x[2][1])
    group = q[0][3]
    cluster = []
    for i in range(len(q)):
        print(i)
        if q[i][3] !=group:
            break
        cluster.append(q[i])
    dict_cluster_word = {}
    dict_cluster_word['center_word'] = compare_word
    dict_cluster_word['Max'] = max([i[2][1] for i in cluster])
    dict_cluster_word['Min'] = min([i[2][1] for i in cluster])
    dict_cluster_word['list_word'] = str([i[1] for i in cluster])
    print (dict_cluster_word)
    conn = sqlite3.connect('database_similarity.db')
    c = conn.cursor()
    cluster_for_insert = [(dict_cluster_word['center_word'],dict_cluster_word['Max'],dict_cluster_word['Min'],dict_cluster_word['list_word'])]
    c.executemany("INSERT INTO cluster_word VALUES (?,?,?,?)",cluster_for_insert)
    # c.execute("SELECT * from cluster_word")
    # print(c.fetchall())
    conn.commit()
    conn.close()



    # fi.close()
    # cluster_queue.put(1)


def main():
    global list_of_word_global
    # a = ['a','the','because','mother',';']
    # print (remove_stopwords(a))
    # return 0

    # return 0
    print(os.getpid())
    list_name_of_file = []
    # for i in range(100):
    #     if i%2 == 0:
    #         list_name_of_file+=['pg25990.txt']
    #     else:
    #         list_name_of_file+=['pg706.txt']
    #
    # list_name_of_file = ['pg25990.txt']#,'pg706.txt']
    # read_from_file(list_name_of_file[0])

    pool = mp.Pool(mp.cpu_count())
    # pool.map(read_from_file,list_name_of_file)
    list_of_document = read_from_file_csv('datasetsmall.csv')
        # list_name_of_file = read_from_file_csv('dataset.csv')
    print(len(list_of_document))
    pool.map(read_from_file,list_of_document)


    list_of_words = []
    # for i in range(len(list_name_of_file)):
    for i in range(len(list_of_document)):
        list_of_words+= queue_list_word.get()
        list_of_words = uniq_word(list_of_words)
        # print(len(list_of_words))
    print(len(list_of_words))

    pool.close()
    # return 0
    pool = mp.Pool(mp.cpu_count())
    list_of_word_global = list_of_words
    list_for_calculate_distance = [(i,list_of_words[i],list_of_words) for i in range(len(list_of_words))]

    pool.map(calculate_distance,list_for_calculate_distance)
    temp = []
    for i in range(len(list_for_calculate_distance)):
        temp.append(distance_queue.get())

    pool.close()
    temp = sorted(temp,key =lambda x:x[0])
    # print (temp[0][2])
    # for i in temp:
    #     print (i[2][1])
    word_distance = [i[2] for i in temp]
    list_for_kmean_clustering = [(list_of_words[i],word_distance[i]) for i in range(len(word_distance))]
    time.sleep(10.0)
    # pool = mp.Pool(2)
    # for j in
    # pool.map(kmean_clustering,list_for_kmean_clustering)
    for i in list_for_kmean_clustering:
        kmean_clustering(i)
    # pool = mp.Pool(2)
    # pool.map(kmean_clustering,list_for_kmean_clustering)
    # for i in range(len(list_for_kmean_clustering)):
    #     cluster_queue.get()

    # print (list_for_kmean_clustering)
    # print(len(list_for_kmean_clustering))
    # pool.map(kmean_clustering,list_for_kmean_clustering)
    # pool.close()
    # for i in range(len(list_for_kmean_clustering)):
    #     temp_queue.get
    # kmean_clustering(list_for_kmean_clustering)
    # for k in range(0,2):
    #     f = []
    #     z = []
    #     # print (l[k])
    #     for j in range(0,len(l)):
    #         w = stringdist(list_of_words[k] ,list_of_words[j],method = 'cosine',q = 2)
    #         if type(w[1]) is not str:
    #             z.append([0,w[1]])
    #         else:
    #             z.append((l[i],l[j],12321))
    #     f = z
    #
    #     # mat = np.matrix(f)
    #     X = np.asarray(f)
    #
    #
    #     range_n_clusters = range(2,100)
    #     # print (range_n_clusters)
    #     Max = 0
    #     index = 0
    #     Z = 0
    #     # fi = open('./KMEAN/'+str(l[k]),'w+')
    #     for n_clusters in range_n_clusters:
    #
    #         clusterer = KMeans(n_clusters=n_clusters ,init='k-means++')
    #         cluster_labels = clusterer.fit_predict(X)
    #         silhouette_avg = silhouette_score(X, cluster_labels)
    #         if Max < silhouette_avg:
    #             Max = silhouette_avg
    #             index = n_clusters
    #             Z = cluster_labels
    #             # print (index,count)
    #         if Max == 1:
    #             break
    #
    #         # fi.write("For n_clusters = "+str(n_clusters)+
    #         #       " The average silhouette_score is : "+str(silhouette_avg)+'\n')
    #         # count+=1
    #     # fi.write(str(index)+"\n")
    #     # q = [(l[i],f[i],Z[i]) for i in range(len(Z))]
    #     # q = sorted(q,key=operator.itemgetter(1,0))
    #     # for i in range(len(q)):
    #     #     fi.write(str(q[i])+'\n')
    #     # fi.close()



if __name__ == "__main__":

    main()
