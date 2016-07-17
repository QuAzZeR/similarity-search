# from __future__ import print_function
# from sklearn.datasets import make_blobs
# from sklearn.cluster import SpectralClustering
# from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from edit_distance import stringdist
from nltk.corpus import stopwords
import math
import nltk
import string
import numpy as np
import multiprocessing as mp



stop_words = set(stopwords.words('english'))
stops_add = set(["'d", "'ll", "'m", "'re", "'s", "'t", "n't", "'ve"])
stop_words = stop_words.union(stops_add)
punctuations = set(string.punctuation)
punctuations.add("''")
punctuations.add("``")
stops_and_punctuations = punctuations.union(stop_words)
queue_list_word = mp.Queue()
def uniq_word(list_non_uniq):
    return list(set(list_non_uniq))

def read_from_file(args):
    file_name = args
    print(file_name)
    # print
    open_file = open('../Data/'+file_name,'r')
    list_of_words = get_word_from_text(open_file)
    open_file.close()
    queue_list_word.put(list_of_words)

def get_word_from_text(open_file):
    list_of_words = nltk.word_tokenize(open_file.read())
    list_of_words = uniq_word(list_of_words)
    list_of_words = remove_stopwords(list_of_words)
    return list_of_words

def remove_stopwords(list_of_words):
    return [i for i in list_of_words if i not in stops_and_punctuations]

def kmean_clustering(args):
    f = []
    z = []
    # print (l[k])
    compare_word = args[0]
    list_of_words = args[1]

    for j in range(0,len(list_of_words)):
        w = stringdist(list_of_words[k] ,list_of_words[j],method = 'cosine',q = 2)
        if type(w[1]) is not str:
            z.append([0,w[1]])
        else:
            z.append((l[i],l[j],12321))
    f = z

    # mat = np.matrix(f)
    X = np.asarray(f)


    range_n_clusters = range(2,100)
    # print (range_n_clusters)
    Max = 0
    index = 0
    Z = 0
    # fi = open('./KMEAN/'+str(l[k]),'w+')
    for n_clusters in range_n_clusters:

        clusterer = KMeans(n_clusters=n_clusters ,init='k-means++')
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        if Max < silhouette_avg:
            Max = silhouette_avg
            index = n_clusters
            Z = cluster_labels
            # print (index,count)
        if Max == 1:
            break

        # fi.write("For n_clusters = "+str(n_clusters)+
        #       " The average silhouette_score is : "+str(silhouette_avg)+'\n')
        # count+=1
    # fi.write(str(index)+"\n")
    # q = [(l[i],f[i],Z[i]) for i in range(len(Z))]
    # q = sorted(q,key=operator.itemgetter(1,0))
    # for i in range(len(q)):
    #     fi.write(str(q[i])+'\n')
    # fi.close()

def main():
    # a = ['a','the','because','mother',';']
    # print (remove_stopwords(a))
    # return 0
    list_name_of_file = []
    for i in range(100):
        if i%2 == 0:
            list_name_of_file+=['pg25990.txt']
        else:
            list_name_of_file+=['pg706.txt']


    pool = mp.Pool(mp.cpu_count())
    pool.map(read_from_file,list_name_of_file)
    list_of_words = []
    for i in range(len(list_name_of_file)):
        list_of_words+= queue_list_word.get()
        list_of_words = uniq_word(list_of_words)
        print(len(list_of_words))


    pool = mp.Pool(mp.cpu_count())
    list_for_kmean_clustering = [(i,list_of_words) for i in list_of_words]
    print(len(list_for_kmean_clustering))
    #
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
