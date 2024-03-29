import numpy as np
import networkx as nx
from scipy.spatial import distance
from utils.preprocess import preprocess
from chinese_whispers import chinese_whispers, aggregate_clusters

def make_graph(comments_vec):
    G = nx.Graph()

    for i, comment in enumerate(comments_vec):
        G.add_node(str(i))
        for ii, other_comment in enumerate(comments_vec[i:]):
            G.add_node(str(ii))
            ii += i 
            if i != ii:
                similarity = 1- distance.cosine(comment, other_comment)
                node_1 = str(i)
                node_2 = str(ii)
                if similarity >= 0.13:
                    # print(similarity)
                    G.add_edge(node_1, node_2, weight=(similarity*100)**2) 
    return G


def get_most_popular(comments: list):
    aggregate_comments = ' '.join(comments)
    single_words = aggregate_comments.split(' ')
    bow = set(single_words)

    most_popular_words = {}

    for fw in bow:
        most_popular_words[fw] = 0
        for c in comments:
            counted_flag = 0
            for w in c.split(' '):
                if ((fw == w) and (counted_flag == 0)):
                    most_popular_words[fw] += 1
                    counted_flag = 1

    sorted_popular_words = [(k,v) for k, v in sorted(most_popular_words.items(), key=lambda item: item[1], reverse=True)]
    return sorted_popular_words

    
if __name__ == "__main__":
    from utils.api_scraper import scraper
    import time 

    url = "https://www.youtube.com/watch?v=UFxqMPmxsDA"
    api_key = "AIzaSyC5HZxK4bznwBldhwF_gJXodOqYurYlFqI"
    
    startTime = time.time() # time scrape
    
    s = scraper(api_key)
    comments = s.get_comments(url)

    endTime = time.time() # end time scrape
    print("Found " + str(len(comments)) + " commments in " + format(endTime - startTime,".3f") + " seconds")

    startTime = time.time() # time preprocess

    vec, feature_names, preprocessed_comments, id_map = preprocess(comments, vec='tfidf')

    endTime = time.time() # end time preprocess
    print("Preprocessed in " + format(endTime - startTime,".3f") + " seconds")

    startTime = time.time() # time chinese whispers

    G = make_graph(vec)
    chinese_whispers(G, weighting='top', iterations=2, seed=123)

    endTime = time.time() # end time chinese whispers
    print("Clustering completed in " + format(endTime - startTime,".3f") + " seconds")

    comment_ids = [(preprocessed_comments[i], id_map[i]) for i in range(len(preprocessed_comments))]

    for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
        
        clustered_comments = [comment_ids[int(idx)][0] for idx in cluster]
        raw_clustered_comments = [comments[int(comment_ids[int(idx)][1])] for idx in cluster]
        
        print('\nCluster {} - Contains {} comments  \n'.format(label, len(clustered_comments)))

        print("Processed comments...")
        print(clustered_comments)
        
        print("\nUnprocessed comments...")
        print(raw_clustered_comments)
        
        print()

        most_popular = get_most_popular(clustered_comments)
        print(most_popular[:5])














        ## with pickled comments

        # import pickle
        # from pathlib import Path

        # wdir = Path(__file__).resolve().parents[0]
        # save_dir = wdir / 'utils' / "response.pickle"

        # with open(save_dir, 'rb') as handle:
        #     data = pickle.load(handle)
        #     comments = [i['snippet']['topLevelComment']['snippet']['textDisplay'] for i in data['items']]
        
        #     vec, feature_names, preprocessed_comments = preprocess(comments, vec='tfidf')

        #     G = make_graph(vec)
        #     chinese_whispers(G, weighting='top', seed=123)

        #     nodes = G.nodes(data=True)

        #     for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
        #         print('Class {}'.format(label))

        #         clustered_comments = [preprocessed_comments[int(idx)] for idx in cluster]
        #         print(clustered_comments)