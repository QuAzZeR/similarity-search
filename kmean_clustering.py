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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from edit_distance import stringdist
from kmean_lib import kmeans
# from temp import kmeans

# import numpy as np

# from sklearn.cluster import KMeans


print(__doc__)


def find_max_diff(a,b,maximum,index,max_index,threshold):
    DELTA =0.00001
    if abs(b-a) == 0.0 and max_index  == index-1:
        return maximum,i,b
    elif maximum > abs(b-a):
        return maximum,max_index,threshold
    else:
        return abs(b-a),i,b


def getkey(item):
    return item[1]

def uniq_list(item):
    return list(set(item))

def calculate_threshold(type_method,list_of_word,search_word,k):
    list_of_distance = []
    # list_of_cant_compute_distance = []
    for i in range(len(list_of_word)):
        w = stringdist(search_word ,list_of_word[i],method = type_method,q = 1)
        # print (w,type(w[1]),end='')
        # print (type(w[1] )) =
        # print(w)
        if type(w[1]) is not str:
            list_of_distance.append(1-w)
            # print(2)
            # list_of_cant_compute_distance.append(w)

    # print(list_of_distance)


    list_of_distance = uniq_list(list_of_distance)
    list_of_distance = sorted(list_of_distance,key = getkey)
    # list_of_cant_compute_distance = uniq_list(list_of_cant_compute_distance)
    z = [x for x in list_of_distance]
    # print z
    try:
        return kmeans(z,k,10),1
    except:
        return 0,0
#
#
# f = open('./Log/'+type_method+'_distance_'+search_word,'w+')
# for i in range(len(list_of_distance)):
#     f.write(str(list_of_distance[i][0])+" "+str(list_of_distance[i][1])+"\n")
#     # if i == index:
# #         f.write("---------------------------\n")
# for i in range(len(list_of_cant_compute_distance)):
#     f.write(str(list_of_cant_compute_distance[i][0])+" "+str(list_of_cant_compute_distance[i][1])+"\n")
# f.close()
# #     # pass

f = open('../Data/pg25990.txt','r')
# f = open('./t','r')
list_of_word = []
# list_of_word = nltk.word_tokenize(f.read())
z = f.read().split('\n')
# print z
f.close()

for i in range(len(z)):
        list_of_word += z[i].split()
# print len(list_of_word)
list_of_word = list(set(list_of_word))
# print len(list_of_word)
l = list_of_word

for k in range(0,2):
    f = []
    z = []
    print (l[k])
    for j in range(0,len(l)):
        w = stringdist(l[k] ,l[j],method = 'cosine',q = 2)
        if type(w[1]) is not str:
            z.append([0,w[1]])
        else:
            z.append((l[i],l[j],12321))
    f = z

    mat = np.matrix(f)
    X = np.asarray(f)
    # for i in z:
    #     print (i[1])

    range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16]#,17,18,19,20]
    Max = 0
    index = 0
    Z = 0
    fi = open('./KMEAN/'+str(l[k]),'w+')
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        # ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        # ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        # eigen_values, eigen_vectors = np.linalg.eigh(X)
        # print (eigen_vectors)
        clusterer = KMeans(n_clusters=n_clusters ,init='k-means++')
        # print (clusterer)
        cluster_labels = clusterer.fit_predict(X)
        # cluster_labels = clusterer.fit_predict(eigen_vectors[:, 2:4])
        # print("Length = "+str(len(cluster_labels)))
        # for i in range(len(cluster_labels)):
        #     print (cluster_labels[i])
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        if Max < silhouette_avg:
            Max = silhouette_avg
            index = n_clusters
            Z = cluster_labels
            # print (index,count)

        fi.write("For n_clusters = "+str(n_clusters)+
              " The average silhouette_score is : "+str(silhouette_avg)+'\n')
        # count+=1
    fi.write(str(index)+"\n")
    q = [(l[i],f[i],Z[i]) for i in range(len(Z))]
    q = sorted(q,key=operator.itemgetter(1,0))
    for i in range(len(q)):
        fi.write(str(q[i])+'\n')
    fi.close()
    # print(l)
    # calculate_threshold('cosine',l,'other',index)
        # Compute the silhouette scores for each sample
        # sample_silhouette_values = silhouette_samples(X, cluster_labels)
        #
        # y_lower = 10
        # for i in range(n_clusters):
        #     # Aggregate the silhouette scores for samples belonging to
        #     # cluster i, and sort them
        #     ith_cluster_silhouette_values = \
        #         sample_silhouette_values[cluster_labels == i]
        #
        #     ith_cluster_silhouette_values.sort()
        #
        #     size_cluster_i = ith_cluster_silhouette_values.shape[0]
        #     y_upper = y_lower + size_cluster_i
        #
        #     color = cm.spectral(float(i) / n_clusters)
        #     ax1.fill_betweenx(np.arange(y_lower, y_upper),
        #                       0, ith_cluster_silhouette_values,
        #                       facecolor=color, edgecolor=color, alpha=0.7)
        #
        #     # Label the silhouette plots with their cluster numbers at the middle
        #     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        #
        #     # Compute the new y_lower for next plot
        #     y_lower = y_upper + 10  # 10 for the 0 samples
        #
        # ax1.set_title("The silhouette plot for the various clusters.")
        # ax1.set_xlabel("The silhouette coefficient values")
        # ax1.set_ylabel("Cluster label")
        #
        # # The vertical line for average silhoutte score of all the values
        # ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        #
        # ax1.set_yticks([])  # Clear the yaxis labels / ticks
        # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        #
        # # 2nd Plot showing the actual clusters formed
        # colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
        #             c=colors)
        #
        # # Labeling the clusters
        # centers = clusterer.cluster_centers_
        # # Draw white circles at cluster centers
        # ax2.scatter(centers[:, 0], centers[:, 1],
        #             marker='o', c="white", alpha=1, s=200)
        #
        # for i, c in enumerate(centers):
        #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
        #
        # ax2.set_title("The visualization of the clustered data.")
        # ax2.set_xlabel("Feature space for the 1st feature")
        # ax2.set_ylabel("Feature space for the 2nd feature")
        #
        # plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
        #               "with n_clusters = %d" % n_clusters),
        #              fontsize=14, fontweight='bold')
        #
        # plt.show()
