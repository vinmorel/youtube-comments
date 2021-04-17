# implement main pipeline here with tools from /utils
import getopt
import pickle
import re
import sys
from os import path
from pathlib import Path

import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

from utils.api_scraper import scraper
from utils.preprocess import preprocess
from utils.gather_results import saver

from chinese_whisp import make_graph, get_most_popular

import time


if __name__ == "__main__":

    videoURL = ""
    apiKey = ""
    prepType = ""
    clusterType = ""
    noClusters = 10

#extract commmand line arguments
    helpMsg = "Usage: -y <YouTube video URL> -k <API Key> -p <t | b = TF-IDF, BOW> -c <k | d | s | l | cw = KMeans, DBSCAN, Spectral, LDA, ChineseWhispers> -n <Number of clusters>"

    if len(sys.argv) == 1:
        print(helpMsg)
        sys.exit()

    try:
        options, args = getopt.getopt(sys.argv[1:], "y:k:p:c:n:h", ["videoURL =", "apiKey =", "prepType =", "clusterType =", "noClusters ="])
    except:
        print(helpMsg)
    
    for name, value in options:
        if name == '-y':
            videoURL = value.strip()
        elif name == '-k':
            apiKey = value.strip()
        elif name == '-p':
            prepType = value.strip()
        elif name == '-c':
            clusterType = value.strip()
        elif name == '-n':
            noClusters = value.strip()
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

    if clusterType not in ['k', 'd', 's', 'l', 'cw']:
        print("Invalid cluster type, using K (KMeans)...")
        clusterType = 'k'

    try:
        s = videoURL.find('v=') + 2
        fileName = videoURL[s : s + 15]
    except:
        fileName = "No name"

#get comments file

    #if available use previous saved pickle file

    startTime = time.time()

    if path.exists(fileName + ".pickle"):
        print("Use previous extracted commments from " + fileName + ".pickle" + "...")

        with open(fileName + ".pickle", 'rb') as handle:
            data = pickle.load(handle)
            original_comments = []
            for comment in data:
                for item in comment['items']:
                    original_comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
    else:
        print("Extracting commments from " + videoURL + "...")

        s = scraper(apiKey)
        original_comments = s.get_comments(videoURL, save_to_disk=True, pickle_name=fileName + ".pickle")

    endTime = time.time()

    print("Found " + str(len(original_comments)) + " commments in " + format(endTime - startTime,".3f") + " seconds")

#preprocess comments
    map_id = []

    startTime = time.time()

    if prepType == 't':
        # tfidf
        print("Preprocessing " + str(len(original_comments)) + " commments by TD-IDF...")    
        preprocessed_comments, feature_names, proc_comments, map_id = preprocess(original_comments, vec='tfidf')
    elif prepType == 'b':
        # bag of words
        print("Preprocessing " + str(len(original_comments)) +" commments by Bag of Words...")
        preprocessed_comments, feature_names, proc_comments, map_id = preprocess(original_comments, vec='bow')
    
    endTime = time.time()
    print("Preprocessed in " + format(endTime - startTime,".3f") + " seconds")

#cluster comments
    startTime = time.time()

    if clusterType == 'k':
        print("Clustering with KMeans in " + str(NO_CLUSTERS) + " clusters...")
        endTime = time.time()
        clust = KMeans(n_clusters=NO_CLUSTERS, random_state=0).fit(preprocessed_comments)
        list_clusters = clust.labels_
    elif clusterType == 'd':
        print("Clustering with DBSCAN...")
        clust = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(preprocessed_comments)  #jaccard not supported with sparse matrix
        list_clusters = clust.labels_
    elif clusterType == 's':
        print("Clustering with SpectralClustering in " + str(NO_CLUSTERS) + " clusters...")
        clust = SpectralClustering(n_clusters=NO_CLUSTERS, assign_labels="discretize", random_state=0).fit(preprocessed_comments)
        list_clusters = clust.labels_
    elif clusterType == 'l':
        print("Clustering with LDA in " + str(NO_CLUSTERS) + " clusters...")

        comments_tokenized = [simple_preprocess(doc) for doc in proc_comments]
        dictionary = corpora.Dictionary()
        bow_corpus = [dictionary.doc2bow(comment, allow_update=True) for comment in comments_tokenized]
        #print(bow_corpus)

        if prepType == 't':
            # tfidf
            tfidf = models.TfidfModel(bow_corpus)
            corpus_tfidf = tfidf[bow_corpus]
            lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=NO_CLUSTERS, id2word=dictionary, passes=2, workers=4)
        else:
            lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=NO_CLUSTERS, id2word=dictionary, passes=2, workers=4)
        
        #print(lda_model.print_topics(-1, 5))

        list_clusters = lda_model.print_topics(-1)
    elif clusterType == 'cw':
        import os
        from chinese_whispers import chinese_whispers, aggregate_clusters
        import numpy as np
        import networkx as nx
        from scipy.spatial import distance

        print("Clustering with Chinese Whispers...")
        
        G = make_graph(preprocessed_comments)
        chinese_whispers(G, weighting='top', iterations=2, seed=123)

        comment_ids = [(proc_comments[i], map_id[i]) for i in range(len(proc_comments))]

        fdir = Path(__file__).resolve().parents[0]

        lab = 0

        with open('ChineseWhispOut.txt', 'w', encoding="utf-8") as f:
            for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
                clustered_comments = [comment_ids[int(idx)][0] for idx in cluster]
                raw_clustered_comments = [original_comments[int(comment_ids[int(idx)][1])] for idx in cluster]
                most_popular = get_most_popular(clustered_comments)[:5]

                f.write("----"*60 + '\n')
                f.write('Cluster {} \nNumber of comments : {}  \nMost popular words : {} \n\n'.format(lab, len(clustered_comments), most_popular))

                f.write("Processed comments... \n")
                f.write(str(clustered_comments) + "\n")
                
                f.write("\nUnprocessed comments... \n")
                f.write(str(raw_clustered_comments) + "\n")
                
                f.write('\n\n')

                lab += 1

        os.system("start notepad.exe {}".format(str(fdir)+"\ChineseWhispOut.txt"))

    endTime = time.time()
    print("Clustering completed in " + format(endTime - startTime,".3f") + " seconds")

#save clusters in a text file
    savedFileName = str(fileName) + '_' + str(NO_CLUSTERS) + '_' + clusterType + '_' + prepType + '.txt'
    print("Saving clusters to the file " + savedFileName + "...")

    sa = saver()

    if clusterType == 'l':
        sa.save_topics(savedFileName, list_clusters)
    elif clusterType != 'cw' :
        sa.save_clusters(savedFileName, list_clusters, original_comments, proc_comments, map_id)