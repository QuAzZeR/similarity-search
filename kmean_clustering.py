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



queue_list_word = mp.Queue()
temp_queue = mp.Queue()
lock = mp.Lock()

def uniq_word(list_non_uniq):
    return list(set(list_non_uniq))

def read_from_file(args):
    file_name = args
    # print(file_name)
    # print
    open_file = open('../Data/'+file_name,'r')
    list_of_words = get_word_from_text(open_file)
    open_file.close()
    queue_list_word.put(list_of_words)

def get_word_from_text(open_file):
    list_of_words = nltk.word_tokenize(open_file.read())
    list_of_words = uniq_word(list_of_words)
    list_of_words = remove_stopword(list_of_words)
    return list_of_words



def kmean_clustering(args):

    compare_word = args[0]
    list_of_words = args[1]

    word_distance = []
    for word in list_of_words:
        w = stringdist(compare_word ,word,q = 2)
        word_distance.append((0,w[1]))

    # print(os.getpid())

    print(compare_word+' word_distance  = '+str(len(word_distance)))
    X = np.asarray(word_distance)


    range_n_clusters = range(2,100)
    Max = 0
    index = 0
    Z = 0
    count = 0


    for n_clusters in range_n_clusters:
        print(compare_word+' cluster = '+str(n_clusters))

        clusterer = KMeans(n_clusters=n_clusters ,init='k-means++')
        print(1)
        cluster_labels = clusterer.fit_predict(X)
        print(2)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(3)

        if Max < silhouette_avg:
            Max = silhouette_avg
            index = n_clusters
            Z = cluster_labels
            # print (index,count)
            count = 1

        else:
            count+=1
        if count == 5 or Max == 1:
            break
        print(4)
        print("For n_clusters = "+str(n_clusters)+
              " The average silhouette_score is : "+str(silhouette_avg)+'\n')
        print(5)
        # count+=1
    # fi.write(str(index)+"\n")
    q = [(compare_word,list_of_words[i],word_distance[i],Z[i]) for i in range(len(Z))]
    q = sorted(q,key=operator.itemgetter(1,0))
    for i in range(len(q)):
        print(str(q[i]))
    # fi.close()


def main():
    # a = ['a','the','because','mother',';']
    # print (remove_stopwords(a))
    # return 0
    print(os.getpid())
    list_name_of_file = []
    for i in range(100):
        if i%2 == 0:
            list_name_of_file+=['pg25990.txt']
        else:
            list_name_of_file+=['pg706.txt']

    list_name_of_file = ['pg25990.txt','pg706.txt']
    # read_from_file(list_name_of_file[0])
    pool = mp.Pool(mp.cpu_count())
    pool.map(read_from_file,list_name_of_file)

    list_of_words = []
    for i in range(len(list_name_of_file)):
        list_of_words+= queue_list_word.get()
        list_of_words = uniq_word(list_of_words)
        # print(len(list_of_words))

    pool.close()
    pool = mp.Pool(mp.cpu_count())

    # list_for_kmean_clustering = [(i,list_of_words) for i in list_of_words]
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
