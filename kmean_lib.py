import os
import numpy as np

# kmeans clustering algorithm
# data = set of data points
# k = number of clusters
# c = initial list of centroids (if provided)
#
def kmeans(data1, k, c):
    centroids = []
    name  =data1
    data = np.asarray([data1[i][1] for i in range(len(data1))])
    centroids = randomize_centroids(data, centroids, k)

    old_centroids = [[] for i in range(k)]

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations)):
        iterations += 1

        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(data, centroids, clusters)

        # recalculate centroids
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1


    # print("The total number of data instances is: " + str(len(data)))
    # print("The total number of iterations necessary is: " + str(iterations))
    # print("-----The means of each cluster are: " + str(centroids)+"------")
    # print("The clusters are as follows:")
    # for cluster in clusters:
    # print("Cluster with a size of " + str(len(cluster)) + " starts here:")
    index = 0
    m = 100000000
    for i in range(len(clusters)):
        z = min(np.array(clusters[i]).tolist())
        m = min(z,m)
        # print m
        # print index
        if m in np.array(clusters[i]).tolist():
            index = i
    # print ("min cluster = "+np.array(clusters[index]).tolist())
    cluster = clusters[index]
    for cluster in clusters:
        for i in range(len(cluster)):
            ans,name = find_name(cluster[i],name)
            print (ans,cluster[i])
        print ("-----------------------")
        # print(np.array(clusters[0]).tolist())
    # print("---Cluster ends here.---")
    cluster = clusters[index]
    return max(cluster)

def find_name(value,name):
    for i in range(len(name)):
        if name[i][1] == value:
            ans = name[i][0]
            name.remove((ans,value))
            return ans,name
    return "not found",name

# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.
def euclidean_dist(data, centroids, clusters):
    for instance in data:
        # Find which centroid is the closest
        # to the given data point.
        mu_index = min([(i[0], np.linalg.norm(instance-centroids[i[0]])) \
                            for i in enumerate(centroids)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters


# randomize initial centroids
def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroids


# check if clusters have converged
def has_converged(centroids, old_centroids, iterations):
    MAX_ITERATIONS = 1000
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids

# kmeans(np.asarray([1.012321124,1.0122231312312,1.12312311,1.2,1.3,1.4,3.0,9.0,150.0,13.0,69.0]),3,10)
