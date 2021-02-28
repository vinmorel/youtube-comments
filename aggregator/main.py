# implement main pipeline here with tools from /utils
from utils.preprocess import preprocess
from utils.api_scraper import scraper

import getopt
import sys
import re
from os import path

import pickle
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

if __name__ == "__main__":

    videoURL = ""
    apiKey = ""
    prepType = ""
    clusterType = ""
    noClusters = 10

#extract commmand line arguments
    helpMsg = "Usage: -y <YouTube video URL> -k <API Key> -p <t | b = TF-IDF, BOW> -c <k | d | s = KMeans, DBSCAN, Spectral> -n <Number of clusters>"

    if len(sys.argv) == 1:
        print(helpMsg)
        sys.exit()

    try:
        options, args = getopt.getopt(sys.argv[1:], "y:k:p:c:n:h", ["videoURL =", "apiKey =", "prepType =", "clusterType =", "noClusters ="])
    except:
        print(helpMsg)
    
    for name, value in options:
        if name == '-y':
            videoURL = value
        elif name == '-k':
            apiKey = value
        elif name == '-p':
            prepType = value
        elif name == '-c':
            clusterType = value
        elif name == '-n':
            noClusters = value
        elif name == '-h':
            print(helpMsg)
        else :
            print(helpMsg)

#validate arguments
    try:
        NO_CLUSTERS = int(noClusters)
    except:
        print("Invalid number of clusters, using 10...")
        NO_CLUSTERS = 10

    if prepType not in ['t', 'b']:
        print("Invalid preprocessing type, using t (TF-IDF)...")
        prepType = 't'

    if clusterType not in ['k', 'd', 's']:
        print("Invalid cluster type, using K (KMeans)...")
        clusterType = 'k'

    try:
        s = videoURL.find('v=') + 2
        fileName = videoURL[s : s + 15]
    except:
        fileName = "No name"

#get comments file
    # "https://www.youtube.com/watch?v=7MFKy7DJsCY"   #Lost World of the Maya (Full Episode) | National Geographic
    # error -> check_lang

    #if available use previous saved pickle file

    if path.exists(fileName + ".pickle"):
        print("Use previous extracted commments from " + fileName + ".pickle" + "...")

        with open(fileName + ".pickle", 'rb') as handle:
            data = pickle.load(handle)
            original_comments = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in data['items']]
    else:
        print("Extracting commments from " + videoURL + "...")

        s = scraper(apiKey)
        original_comments = s.get_comments(videoURL, save_to_disk=True, pickle_name=fileName + ".pickle")

#preprocess comments

    if prepType == 't':
        # tfidf
        print("Preprocessing " + str(len(original_comments)) + " commments by TD-IDF...")
        preprocessed_comments, feature_names = preprocess(original_comments, vec='tfidf')
    elif prepType == 'b':
        # bag of words
        print("Preprocessing " + str(len(original_comments)) +" commments by Bag of Words...")
        preprocessed_comments, feature_names = preprocess(original_comments, vec='bow')
    
    # standard 
    proc_comments = preprocess(original_comments)

#cluster comments

    if clusterType == 'k':
        print("Clustering with KMeans in " + str(NO_CLUSTERS) + " clusters...")
        clust = KMeans(n_clusters=NO_CLUSTERS, random_state=0).fit(preprocessed_comments)
    elif clusterType == 'd':
        print("Clustering with DBSCAN...")
        clust = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(preprocessed_comments)  #jaccard not supported with sparse matrix
    elif clusterType == 's':
        print("Clustering with SpectralClustering in " + str(NO_CLUSTERS) + " clusters...")
        clust = SpectralClustering(n_clusters=NO_CLUSTERS, assign_labels="discretize", random_state=0).fit(preprocessed_comments)

    print(clust.labels_)

    orig_clusters={}
    proc_clusters={}

    for i in range(len(clust.labels_)):
        cluster_idx = clust.labels_[i]
        if orig_clusters.get(cluster_idx) == None:
            orig_clusters[cluster_idx] = []
            proc_clusters[cluster_idx] = []
        (orig_clusters[cluster_idx]).append(original_comments[i])
        (proc_clusters[cluster_idx]).append(proc_comments[i])

#save clusters in a text file
    savedFileName = str(fileName) + '_' + str(NO_CLUSTERS) + '_' + clusterType + '.txt'
    print("Saving clusters to the file " + savedFileName + "...")

    with open(savedFileName, 'w', encoding="utf-8") as f:
        f.write("Number of clusters: {0:d} ({1:d} comments in total)\n\n".format(NO_CLUSTERS,len(clust.labels_)))
        for key in orig_clusters.keys():
            f.write("CLUSTER %d (original %d comments):\n%s\n\n"%(key,len(orig_clusters[key]),orig_clusters[key]))
            f.write("CLUSTER %d (processed %d comments):\n%s\n\n\n"%(key,len(proc_clusters[key]),proc_clusters[key]))