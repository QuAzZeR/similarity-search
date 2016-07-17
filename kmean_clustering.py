from __future__ import print_function

import math
import random
import operator
# from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from edit_distance import stringdist
from nltk.corpus import stopwords
import nltk
import string
# from kmean_lib import kmeans

# print(__doc__)

stop_words = set(stopwords.words('english'))
stops_add = set(["'d", "'ll", "'m", "'re", "'s", "'t", "n't", "'ve"])
stop_words = stop_words.union(stops_add)
punctuations = set(string.punctuation)
punctuations.add("''")
punctuations.add("``")
stops_and_punctuations = punctuations.union(stop_words)

def read_file(args):
    f = open('../Data/pg25990.txt','r')
    list_of_word = nltk.word_tokenize(f.read())

def remove_stopwords(list_of_word):
    return [i for i in list_of_word if i not in stops_and_punctuations]

def main():
    a = ['a','the','because','mother']
    print (remove_stopwords(a))
    return 0
    f = open('../Data/pg25990.txt','r')
    # f = open('./t','r')

    # print (stop_words)
    list_of_word = []
    list_of_word = nltk.word_tokenize(f.read())
    # z = f.read().split('\n')
    # print z
    f.close()

    # for i in range(len(z)):
            # list_of_word += z[i].split()
    # print (len(list_of_word))
    list_of_word = list(set(list_of_word))
    # print (len(list_of_word))
    l = list_of_word

    for k in range(0,2):
        f = []
        z = []
        # print (l[k])
        for j in range(0,len(l)):
            w = stringdist(l[k] ,l[j],method = 'cosine',q = 2)
            if type(w[1]) is not str:
                z.append([0,w[1]])
            else:
                z.append((l[i],l[j],12321))
        f = z

        mat = np.matrix(f)
        X = np.asarray(f)


        range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16]#,17,18,19,20]
        Max = 0
        index = 0
        Z = 0
        fi = open('./KMEAN/'+str(l[k]),'w+')
        for n_clusters in range_n_clusters:

            clusterer = KMeans(n_clusters=n_clusters ,init='k-means++')
            cluster_labels = clusterer.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            if Max < silhouette_avg:
                Max = silhouette_avg
                index = n_clusters
                Z = cluster_labels
                # print (index,count)
            if MAX == 1:
                pass
            fi.write("For n_clusters = "+str(n_clusters)+
                  " The average silhouette_score is : "+str(silhouette_avg)+'\n')
            # count+=1
        fi.write(str(index)+"\n")
        q = [(l[i],f[i],Z[i]) for i in range(len(Z))]
        q = sorted(q,key=operator.itemgetter(1,0))
        for i in range(len(q)):
            fi.write(str(q[i])+'\n')
        fi.close()



if __name__ == "__main__":

    main()
