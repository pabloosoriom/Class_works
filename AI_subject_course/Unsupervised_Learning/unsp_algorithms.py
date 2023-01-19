#Implementation of Unsupervised Learning Algorithms
#important packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
import scipy
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.pyplot import legend
from array import array
#Warning
import warnings
warnings.filterwarnings('ignore')

#Normalizing the data
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


#Distances between points
#Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
#Manhattan distance
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))
#Minkowski distance
def minkowski_distance(x1, x2, p):
    return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)
#Cosine distance
def cosine_distance(x1, x2):
    return 1 - np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
#Jaccard distance
def jaccard_distance(x1, x2):
    return 1 - np.sum(np.minimum(x1, x2))/np.sum(np.maximum(x1, x2))
#Hamming distance
def hamming_distance(x1, x2):
    return np.sum(x1 != x2)/len(x1)
#Mahalanobis distance
def mahalanobis_distance(x1, x2, V):
    return np.sqrt(np.dot(np.dot((x1 - x2), np.linalg.inv(V)), (x1 - x2).T))


'''
k-means algorithm
'''
#Initialization


def initialize_centroids(X, k):
    #Randomly choose k data points as initial centroids
    return X[np.random.choice(X.shape[0], k, replace=False)]

#Assignment
def assign_clusters(X, centroids):
    #Compute distances between each data point and each centroid
    distances = scipy.spatial.distance.cdist(X, centroids)
    #Assign data points to the closest centroid
    return np.argmin(distances, axis=1)

#Update
def update_centroids(X, k, clusters):
    #Update centroids by taking the average of the assigned data points
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i, :] = np.mean(X[clusters == i, :], axis=0)
    return centroids


#Convergence
def has_converged(centroids, new_centroids):
    #Check if the centroids have changed
    return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in new_centroids]))

def cost_function(X, centroids, metric: str = 'euclidean'):
    #Compute the cost function
    distances = scipy.spatial.distance.cdist(X, centroids, metric=metric)
    cost = np.sum(np.min(distances, axis=1)**2)
    return cost

#K-means
def kmeans(X, k,metric: str = 'euclidean'):
    #Initialize centroids
    centroids = initialize_centroids(X, k)
    #cost function
    cost = []
    #Iterate until convergence
    while True:
        #Assign clusters
        clusters = assign_clusters(X, centroids)
        #Update centroids
        new_centroids = update_centroids(X, k, clusters)
        #Check for convergence
        #print(cost_function(X, centroids))
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
        cost.append(cost_function(X, centroids))
    return centroids, clusters, cost

#Plot
def plot_kmeans(X, k, centroids, clusters):
    #Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=clusters, s=40)
    #Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c=np.arange(k), s=500)
    plt.show()


'''
Fuzzy c-means algorithm
'''

#Initialization

#Initialize U matrix randomly with values between 0 and 1 and the constraint that each row sums to 1
def initialize_U_matrix(X, k):
    U = np.random.rand(len(X), k)
    U = U/np.sum(U, axis=1).reshape(-1, 1)
    return U

#Calculate c-fuzzy centroids
def calculate_c_fuzzy_centroids(X, U, k, m):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        temp_sum=0
        for j in range(len(X)):
            temp_sum+=np.power(U[j, i], m)*X[j, :]
        centroids[i, :] = temp_sum/np.sum(np.power(U[:, i], m))
    return centroids

#calculate cost function
def cost_function_fuzzy(X, centroids, U, k, m):
    distances = scipy.spatial.distance.cdist(X, centroids)
    cost = np.sum(np.power(U, m)*np.power(distances, 2))
    return cost

#Assignment
def update_U_matrix(X, centroids, U, k, m):
    distances = scipy.spatial.distance.cdist(X, centroids)
    new_U=np.zeros((len(X), k));
    for i in range(len(X)):
        for j in range (k):
            temp_sum=0
            for kc in range(k):
                temp_sum+=np.power(distances[i, j]/distances[i, kc], 2/(m - 1))
            new_U[i, j] = 1/temp_sum
    U=new_U;
    return U

#Fuzzy c-means
def fuzzy_cmeans(X, k, m, tol):
    #Initialize the U matrix
    U = initialize_U_matrix(X, k)
    #Initialize the centroids
    centroids = calculate_c_fuzzy_centroids(X, U, k, m)
    #cost function
    cost = []
    #Iterate until convergence
    while True:
        #Update the U matrix
        U = update_U_matrix(X, centroids, U, k, m)
        #Update the centroids
        new_centroids = calculate_c_fuzzy_centroids(X, U, k, m)
        #Check for convergence
        if np.abs(cost_function_fuzzy(X, centroids, U, k, m) - cost_function_fuzzy(X, new_centroids, U, k, m)) <= tol:
            break
        centroids = new_centroids
        #print(cost_function(X, new_centroids, U, k, m))
        cost.append(cost_function_fuzzy(X, new_centroids, U, k, m))
    return centroids, U, cost

#Plot
def plot_fuzzy_cmeans(X, k, centroids, U):
    #Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(U, axis=1), s=40)
    #Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', c=np.arange(k), s=500)
    plt.show()


'''
Mountain algorithm
'''

def mountain(v,x,sigma):
    return np.sum(np.exp(-euclidean_distance(v,x)**(2) /2*sigma**2))

def mountain_clustering(n: int, gr: int, X: np.ndarray, sigma: array, kn: int):
    '''Where n refers to de dimension, gr the grid divisions, X is the raw data
    to be clustered, sigma is an array of deviations for the mountain density
    function, and kn is the number of centers to be found'''
#Set up grid matrix of n-dimensions (V)

    v_dim=gr*np.ones([1,n])
    M = np.zeros([int(x) for x in v_dim[0]])
    M_r=M.reshape(1,-1)[0]
    cur=np.ones([1,n])
    for i in range(0,n):
        for j in range(0,i+1):
            cur[:,i]=cur[:,i]*v_dim[:,j]
    # max_m=[] #greatest density value
    # max_v=[] #Cluster center position
    center=[]
    max_idx=[]

    for k in range(0,kn):
        max_m = 0
        max_v = 0
        max_i=i
        for i in range(0,int(cur[:,-1][0])):
            #Calculate the vector indexes
            idx=i+1
            dim=np.zeros(len(range(n,0,-1))).tolist()
            for j in range(n-1,0,-1):
                dim[j]=(m.ceil(idx/cur[:,j-1]))
                idx=int(idx-cur[:,j-1]*(dim[j]-1))
            dim[0]=idx
            #Dim is holing the current point index vector
            #but needs to be normalized to the range [0,1]
            v=[d /gr for d in dim]
            #calculate the density of the current point
            if k==0:
                M_r[i]=mountain(v,X,sigma[k])
            else:
                M_r[i]=M_r[i]-M_r[max_idx[k-1]]*np.exp(-euclidean_distance(np.array(v),np.array(center[k-1]))**(2) /2*sigma[k]**2)

            #update the max density and the max density point
            if M_r[i]>max_m:
                max_m=M_r[i]
                max_v=v
                max_i=i
        center.append(max_v)
        max_idx.append(max_i)
        print('Cluster ',k+1,' center: ',max_v)
    distances=scipy.spatial.distance.cdist(center, X).T
    #Asigning a cluster to each point with the minimum distance
    clusters=np.argmin(distances,axis=1)
    return clusters, center


'''
Substractive algorithm
'''

def subtractive_clustering(r_a: float, r_b: float, df, kn:int):
    '''Where r_a s the radium  for the first cluster, and r_b the second radio '''
    #Density matrix
    D = np.zeros(len(df.values))

    for i in range(0,len(df.values)):
        for j in range(0,len(df.values)):
            if i!=j:
                D[i]=D[i]+np.exp(-euclidean_distance(df.values[i],df.values[j])**(2) /(2*r_a)**2)

    #Plotting the density with sns
    # sns.scatterplot(x=df.values[:,0],y=df.values[:,1],hue=D)
    # plt.show()

    #Selecting the cluster centers
    #The cluster centers are the points with the greatest density
    centers=[]

    for k in range(0,kn):
        # sc= plt.scatter(df.values[:,0],df.values[:,1],c=D)
        # plt.colorbar(sc)
        # plt.show()

        if k==0:
            centers.append(np.argmax(D))
        else:
            #Update the density matrix
            for i in range(0,len(df.values)):
                D[i]=D[i]-D[centers[k-1]]*np.exp(-euclidean_distance(df.values[i],df.values[centers[k-1]])**(2) /(2*r_b)**2)
            centers.append(np.argmax(D))

    centers_cords=df.values[centers]
    distances=scipy.spatial.distance.cdist(centers_cords, df.values).T
    #Asigning a cluster to each point with the minimum distance
    clusters=np.argmin(distances,axis=1)

    return clusters, centers_cords



'''
Gravity rule clustering algorithm
'''


def gravity_rule_clustering(kn:int, it:int, G_0:float, p:float, t:int, n:int, df):
    V=np.zeros([kn,n])
    a=np.zeros([kn,n])


    #initialize cluster centroids with random points
    # Z=np.random.rand(kn,n)
    Z=df.values[np.random.randint(0,len(df.values),kn)]

    while t<it:
        #Calculate the distance between each point and each cluster centroid
        distances=scipy.spatial.distance.cdist(Z, df.values).T
        #Asigning a cluster to each point with the minimum distance
        clusters=np.argmin(distances,axis=1)
        #Updating the cluster centroids
        #Calculate the total gravity force applied to the cluster centroid Z_j
        for k in range(0,kn):
            C_j=df.values[clusters==k]
            G=G_0*(1-t/it)
            mi,mj=1,1
            sum=0
            for j in range(0,len(C_j)):
                ri=np.random.rand(1,n)
                Rik=euclidean_distance(Z[k],C_j[j])
                epsilon=0.00001
                sum=sum+((ri/(Rik**p+epsilon))*(C_j[j]-Z[k]))
            a[k]=(G/(len(C_j)+epsilon))*sum
            V[k]=V[k]+a[k]
            Z[k]=Z[k]+V[k]

        t=t+1


    #Plotting the clusters
    plt.scatter(df.values[:,0],df.values[:,1],c=clusters)
    plt.scatter(np.array(Z)[:,0],np.array(Z)[:,1],marker='o', c=np.arange(kn), s=500)
    plt.show()

    return clusters, Z


'''
Agglomerative clustering algorithm
'''
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

def hierarchical_clustering(df: pd.DataFrame, method: str, metric: str, n_clusters: int):
    plt.figure(figsize=(10, 7))
    plt.title("Customer Dendograms")
    dend = shc.dendrogram(shc.linkage(df.values, method='ward'))
    plt.show()

    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=metric, linkage='ward')
    cluster.fit_predict(df.values)
    plt.scatter(df.values[:,0],df.values[:,1], c=cluster.labels_, cmap='rainbow')
    plt.show()
    return cluster.labels_, cluster

'''
Evaluation metrics
'''

#Internal clustering criteria indices
#Silhouette index
#Dunn Index
#Davies-Bouldin Index
#Calinski-Harabasz Index

def findDistance(point1, point2):

    eucDis = 0
    for i in range(len(point1)):
        eucDis = eucDis + (point1[i] - point2[i])**2

    return eucDis**0.5

# Function to calcualte Dunn Index
def calcDunnIndex(points, cluster):

    # points -- all data points
    # cluster -- cluster centroids
    numer = float('inf')
    for c in cluster: # for each cluster
        for t in cluster: # for each cluster
           # print(t, c)
            if (t == c).all(): continue # if same cluster, ignore
            ndis = euclidean_distance(t, c)
           # print('Numerator', numerator, ndis)
            numer = min(numer, ndis) # find distance between centroids

    denom = 0
    for c in cluster: # for each cluster
        for p in points: # for each point
            for t in points: # for each point
                if (t == p).all(): continue # if same point, ignore
                ddis = euclidean_distance(t, p)
            #    print('Denominator', denominator, ddis)
                denom = max(denom, ddis)

    return numer/denom


# # Function to calculate The Ball-Hall Index
def calcBallHallIndex(points, clusters):
    #Number of clusters and number of points
    centers=np.unique(clusters)
    K_ = len(centers)
    n = len(points)
    sum=0
    for k in centers: # for each cluster
        #Barycenter of cluster k
        G_k=np.mean(points[clusters==k],axis=0)
        M=points[clusters==k]
        sum_=0
        #Distance between baricenter and each point in cluster k
        for i in range(0,len(M)):
            sum_=sum_+euclidean_distance(G_k,M[i])**2

        sum+=1/len(M)*sum_
    return (1/(K_))*sum

# # Function to calculate the PBM index
def calcPBMIndex(points, clusters):
    centers=np.unique(clusters)
    K_ = len(centers)
    n = len(points)
    #Calculate all baricenters
    G_k=np.zeros((K_,points.shape[1]))
    it=0
    for k in centers:
        G_k[it]=np.mean(points[clusters==k],axis=0)
        it=it+1
    #Calculate all distances between baricenters
    D=np.zeros((K_,K_))
    for i in range(0,K_):
        for j in range(0,K_):
            D[i,j]=euclidean_distance(G_k[i],G_k[j])
    Db=np.max(D)
    #Distance between points and their baricenter
    sum=0
    it=0
    for k in centers: # for each cluster
        #Barycenter of cluster k
        M=points[clusters==k]
        sum_=0
        #Distance between baricenter and each point in cluster k
        for i in range(0,len(M)):
            sum_=sum_+euclidean_distance(G_k[it],M[i])
        sum=sum+sum_
        it=it+1
    Ew=sum
    #Distance between all points and baricenter
    #Baricenter of all points
    G=np.mean(points,axis=0)
    Et=np.zeros((n,K_))
    for i in range(0,n):
        Et[i]=euclidean_distance(points[i],G)
    Et=np.sum(Et)
    #Calculate the PBM index
    PBM=((1/K_)*(Et/Ew)*Db)**2
    return PBM

# # Function to calculate the Mclain-Rao index
def calcMcClainRaoIndex(points, clusters):
    centers=np.unique(clusters)
    K_ = len(centers)
    n = len(points)
    Sw=0
    Nw=0
    for k in centers:
        #Sum of the distances within each cluster
        M=points[clusters==k]
        sum_=0
        it=0
        for i in range(0,len(M)):
            for j in range(0,len(M)):
                sum_=sum_+euclidean_distance(M[i],M[j])
                it=it+1
        Nw=Nw+it
        Sw=Sw+sum_

    Sb=0
    #Sum of the between-cluster distances
    for i in range(0,K_):
        for j in range(0,K_):
            if i==j: continue
            Mi=points[clusters==i]
            Mj=points[clusters==j]
            for k in range(0,len(Mi)):
                for l in range(0,len(Mj)):
                    Sb=Sb+euclidean_distance(Mi[k],Mj[l])
    Nb=n*(n-1)/2-Nw
    #Calculate the McClain-Rao Index
    MR=(Sw/Nw)/(Sb/Nb)
    return MR

# # Function to calcualte all indices
def calcIndices(points, cluster):
    # points -- all data points
    # cluster -- cluster centroids
    # Calculate Dunn Index
    # dunnIndex = calcDunnIndex(points, cluster)
    McClainRIndx=calcMcClainRaoIndex(points, cluster)
    # Calculate Ball-Hall Index
    ballHallIndex = calcBallHallIndex(points, cluster)
    # Calculate PBM Index
    pbmIndex = calcPBMIndex(points, cluster)
    # print('Dunn Index:', calcDunnIndex(points, cluster))
    # print('Ball-Hall Index:', calcBallHallIndex(points, cluster))
    # print('PBM Index:', calcPBMIndex(points, cluster))
    return McClainRIndx, ballHallIndex, pbmIndex



'''
Embeddings
'''
#With TSNE
from sklearn.manifold import TSNE
def tsne_manifold(df: pd.DataFrame, n_components: int, perplexity: int, learning_rate: int, c= None):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    tsne_results = tsne.fit_transform(df.values)
    if n_components == 2:
        plt.scatter(tsne_results[:,0], tsne_results[:,1], c=c)
        plt.title('TSNE projection')
        plt.show()
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], c=c)
        plt.title('TSNE projection')
        plt.show()

    return tsne_results

#With UMAP
import umap
def umap_projection(data, labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric)
    embedding = reducer.fit_transform(data)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in labels])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection', fontsize=24);