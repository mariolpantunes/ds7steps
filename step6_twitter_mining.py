# coding: utf-8
import csv
import re

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering


CSV_DATASET = "twitter.csv"
N_Features = 100
N_Clusters = 10
Top_K_Features = 5


# Rearrange data for print
def rearrange_data(features, y, X, labels, n_clusters):
    clusters = []
    for i in range(n_clusters):
        clusters.append(([], []))

    for i in range(len(y)):
        clusters[labels[i]][0].append(y[i])

    for i in range(len(clusters)):
        vector = np.array([0.0] * N_Features)
        for user in clusters[i][0]:
            idx = y.index(user)
            vector = vector + X[idx]
        for j in range(Top_K_Features):
            idx = np.argmax(vector)
            vector[idx] = -1.0
            clusters[i][1].append(features[idx][0])
    return clusters


# Pretty print clusters
def pretty_print(clusters):
    for i in range(len(clusters)):
        print("Cluster ", i, ":")
        for user in clusters[i][0]:
            print("\t",user)
        print("Trending words: ", clusters[i][1])
        print()


def main():
    # Step 2 pre-processing

    # Remove small words
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    # Remove numbers
    numbers = re.compile(r'\S*\d\S*')

    # Clean dataset
    corpus = []
    raw_data = []
    with open(CSV_DATASET) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            clean_text = shortword.sub('', numbers.sub('', row['tweet'])).strip()
            raw_data.append((row['user'], clean_text))
            corpus.append(clean_text)
    #print(raw_data)
    #print(corpus)

    # Convert it to TF-IDF vectors
    vectorizer = TfidfVectorizer(min_df=1, strip_accents='unicode', stop_words='english')
    X = vectorizer.fit_transform(corpus)
    #print(X)
    idf = sorted(vectorizer.idf_, key=None, reverse=True)
    features = list(zip(vectorizer.get_feature_names(), idf))[:N_Features]
    #print(features)

    # Create dataset
    d = {}
    for user, data in raw_data:
        for i in range(0, len(features)):
            if features[i][0] in data:
                if user not in d:
                    d[user] = [0.0]*N_Features
                d[user][i] += 1.0
    #print(d)
    #print(len(d))
    X=[]
    y=[]

    for key in d.keys():
        y.append(key)
        X.append(d[key])

    # Convert to numpy arrays
    X = np.array(X)

    #print(X)
    #print(y)

    # Learn a model
    # Cluster results using k-means
    kmeans = KMeans(n_clusters=N_Clusters, init='k-means++').fit(X)

    labels = kmeans.labels_
    #centroids = kmeans.cluster_centers_
    #print(labels)
    #print(centroids)

    clusters = rearrange_data(features, y, X, labels, N_Clusters)
    pretty_print(clusters)

    #db = DBSCAN().fit(X)
    #labels = db.labels_
    #n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    #print(n_clusters)
    #clusters = rearrange_data(features, y, X, labels, n_clusters)
    #pretty_print(clusters)

    #af = AffinityPropagation().fit(X)
    #cluster_centers_indices = af.cluster_centers_indices_
    #labels = af.labels_
    #n_clusters = len(cluster_centers_indices)

    #clusters = rearrange_data(features, y, X, labels, n_clusters)
    #pretty_print(clusters)

    #sc = SpectralClustering(n_clusters=N_Clusters).fit(X)
    #labels = sc.labels_
    #clusters = rearrange_data(features, y, X, labels, N_Clusters)
    #pretty_print(clusters)

if __name__ == "__main__":
    main()
