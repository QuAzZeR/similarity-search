from edit_distance import stringdist
# from temp import kmeans
import numpy as np
import math
import random
# import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import operator
from sklearn.cluster import MeanShift, estimate_bandwidth
import csv
import nltk
def find_center_word(l):
    min_avg,max_distance,min_distance = find_avg_distance(l[0],l)
    index_max_avg = 0
    # max_distance = 0
    # min_distance = 0

    for i in range(len(l)):
        avg,max,min= find_avg_distance(l[i],l)
        if min_avg < avg:
            min_avg = avg
            index_max_avg = i
            max_distance = max
            min_distance = min
    return l[index_max_avg],max_distance,min_distance

def find_avg_distance(search_word,l):
    sum = 0
    # z = []
    Min= 1000
    Max= -1
    for j in range(len(l)):
        w = stringdist(search_word.lower(),l[j].lower(),method = 'cosine',q = 2)
        distance = 1-w[1]
        sum+=distance
        # z.append(w[1])
        if Max< distance:
            Max= distance
        if Min> distance:
            Min= distance
    if int(Min) == 1 and int(Max) == 1:
        Min = 0.0
    # print max,min
    return sum/len(l),Max,Min



# def find_max_diff(a,b,maximum,index,max_index,threshold):
#     DELTA =0.00001
#     if abs(b-a) == 0.0 and max_index  == index-1:
#         return maximum,i,b
#     elif maximum > abs(b-a):
#         return maximum,max_index,threshold
#     else:
#         return abs(b-a),i,b


# def getkey(item):
#     return item[1]
#
# def uniq_list(item):
#     return list(set(item))

# def calculate_threshold(type_method,search_word):
#     list_of_distance = []
#     # list_of_cant_compute_distance = []
#     for i in range(len(list_of_word)):
#         w = stringdist(search_word ,list_of_word[i],method = type_method,q = 2)
#         # print (w,type(w[1]),end='')
#         # print (type(w[1] )) =
#         if type(w[1]) is not str:
#             list_of_distance.append(w)
#             # print(2)
#             # list_of_cant_compute_distance.append(w)
#
#     # print(list_of_distance)
#
#
#     list_of_distance = uniq_list(list_of_distance)
#     list_of_distance = sorted(list_of_distance,key = getkey)
#     list_of_cant_compute_distance = uniq_list(list_of_cant_compute_distance)
#     z = [x for x in list_of_distance]
#     # print z
#     try:
#         return kmeans(z,6,10),1
#     except:
#         return 0,0
# #
# #
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
list_of_word = nltk.word_tokenize(f.read())
# z = f.read().split('\n')
# print z
# f.close()
split = 0.67
count = 0
# for i in range(len(z)):
#     print z[i].split()
#     list_of_word += z[i].split()
list_of_word.append('mayer')
list_of_word.append('maier')
print (len(list_of_word  ))
list_of_word = list(set(list_of_word))
# print len(list_of_word)
l = list_of_word
print (len(l))
# l = sorted(l)
f = []
for i in range(len(l)):
    z = []
    for j in range(len(l)):
        w = stringdist(l[i].lower(),l[j].lower(),method = 'cosine',q = 2)
        if type(w[1]) is not str:
            z.append(1-w[1])
        else:
            # print w
            z.append((l[i],l[j],12321))
    f.append(z)
mat = np.matrix(f)
minpts = 1
epsilon = 200
# for epsilon in range(350,500,10):
    # if epsilon >=200 and epsilon <= 250:
    #     continue
z = DBSCAN(eps =epsilon/100.0,min_samples=minpts).fit_predict(mat)
q = [(l[i],z[i]) for i in range(len(z))]
q = sorted(q,key=operator.itemgetter(1,0))
Dict_word = {}
list_of_cluster = []
list_of_cluster.append(q[0][0])
for i in range(1,len(q)):

    if q[i][1]!=q[i-1][1]:
        index_of_dict = q[i-1][1]
        # print find_center_word(list_of_cluster)
        Dict_word[index_of_dict] = {'word' : [],'word_center' : 0,'max' : 0, 'min' : 0}
        Dict_word[index_of_dict]['word_center'],Dict_word[index_of_dict]['max'],Dict_word[index_of_dict]['min'] = find_center_word(list_of_cluster)
        Dict_word[index_of_dict]['word'] = str(list_of_cluster)
        list_of_cluster = []
        list_of_cluster.append(q[i][0])
    else:
        list_of_cluster.append(q[i][0])

import mysql.connector

cnx = mysql.connector.connect(user='root', password='',
                              host='localhost',
                              database='similarity_search')
cursor = cnx.cursor()

# insert_book = "Insert into book (book_name,text) VALUES(%(name_of_book)s,%(text)s)"
# file_name = 'pg25990.txt'
# f = open('../Data/'+file_name,'r')

# book = {'name_of_book':file_name,'text':f.read()}

# cursor.execute(query,book)

# cnx.commit()

insert_cluster_of_book = "Insert into cluster_of_word(book_id,center_word,max_distance,min_distance,similarity_word) VALUE(3,%(word_center)s,%(max)s,%(min)s,%(word)s)"

for i in Dict_word:
    cursor.execute(insert_cluster_of_book,Dict_word[i])
    cnx.commit()
cnx.close()
# lsdf = []
# for i in z:
#     if i not in lsdf:
#         lsdf.append(i)
# print lsdf
'''
f= open('./DBSCAN/NEW1/DBSCANTOKEN/dbscanTOKEN'+str(minpts)+'eps'+str(epsilon),'w')
for i in range(len(z)):
    f.write(str(q[i])+'\n')
f.close()
print "finish eps = "+str(epsilon)+", minpts = "+str(minpts)
'''
# eigen_values, eigen_vectors = np.linalg.eigh(mat)
# z = KMeans(n_clusters=len(l), init='k-means++').fit_predict(eigen_vectors[:, 2:4])
# q = [(l[i],z[i]) for i in range(len(z))]
# q = sorted(q,key=operator.itemgetter(1,0))
# for i in range(len(f)):
#     print q[i]


# for i in range(len(l)):
#     z = []
#     for j in range(i,len(l)):
#         # print l[i],l[j]
#         w = stringdist(l[i] ,l[j],method = 'cosine',q = 2)
#         if type(w[1]) is not str:
#             z.append((l[i],l[j],1-w[1]))
#         else:
#             # print w
#             z.append((l[i],l[j],12321))
#     f.append(z)
# for i in range(len(f)):
#     for j in range(len(f[i])):
#         print str(f[i][j][0])+" "+str(f[i][j][1])+" "+str(f[i][j][2])
# # # for i in f:
# # #     st=''
# # #     for j in i:
# # #         st += str(j)+" "
# # #     print st
# #
# #
# #
# # #-----    -----    -----    -----    -----    -----    -----    -----
# #
# # # print("this book have",len(list_of_word),"words")
# mat = np.matrix(f)
# # eigen_values, eigen_vectors = np.linalg.eigh(mat)
# # z = KMeans(n_clusters=1200, init='k-means++').fit_predict(eigen_vectors[:, 2:4])
# # q = [(l[i],z[i]) for i in range(len(z))]
# # q = sorted(q,key=operator.itemgetter(1,0))
# # # for i in range(len(f)):
# # #     for j in range(len(f[i])):
# # #         print f[i][j]
# #
# # # eigen_values, eigen_vectors = np.linalg.eigh(mat)
# # # z = KMeans(n_clusters=100, init='k-means++').fit_predict(eigen_vectors[:, 2:4])
# # # q = [(l[i],z[i]) for i in range(len(z))]
# # # q = sorted(q,key=operator.itemgetter(1,0))
# # #
# # # for i in range(len(z)):
# # #     print q[i]
# # # print "--------------------------"
# # z = DBSCAN(eps =2,min_samples=1).fit_predict(mat)
# # q = [(l[i],z[i]) for i in range(len(z))]
# # q = sorted(q,key=operator.itemgetter(1,0))
# # lsdf = []
# # for i in z:
# #     if i not in lsdf:
# #         lsdf.append(i)
# # print lsdf
# for i in range(len(z)):
#     print q[i]
# #
# # # for i in range(len(list_of_word)):
# # #     print(list_of_word[i])
# # # s = 0
# # # c = 0.0
# #
# # # type_method = raw_input("what is your method : ")
# # # search_word = raw_input("what is you word : ")
# #
# # # print("Your word is : ",search_word)
# # # print("type of now is : ",type_method)
