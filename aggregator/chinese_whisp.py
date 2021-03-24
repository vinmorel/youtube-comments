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

    
if __name__ == "__main__":
    from utils.api_scraper import scraper
    
    url = "https://www.youtube.com/watch?v=0bt0SjbS3xc&ab_channel=deeplizard"
    api_key = "AIzaSyC5HZxK4bznwBldhwF_gJXodOqYurYlFqI"
    
    s = scraper(api_key)
    comments = s.get_comments(url)

    vec, feature_names, preprocessed_comments, id_map = preprocess(comments, vec='tfidf')

    print(vec)

    G = make_graph(vec)
    chinese_whispers(G, weighting='top', iterations=2, seed=123)

    comment_ids = [(preprocessed_comments[i], id_map[i]) for i in range(len(preprocessed_comments))]

    for label, cluster in sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True):
        print('Cluster {}'.format(label))

        print("Processed comments...")
        clustered_comments = [comment_ids[int(idx)][0] for idx in cluster]
        print(clustered_comments)
        
        print("\nUnprocessed comments...")
        raw_clustered_comments = [comments[int(comment_ids[int(idx)][1])] for idx in cluster]
        print(raw_clustered_comments)
        
        print()
















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