
class saver():

    def save_clusters(self, file_name: str, list_clusters: list, original_comments: list, proc_comments: list, map_id: list):
        
        orig_clusters={}
        proc_clusters={}

        for i in range(len(list_clusters)):
            cluster_idx = list_clusters[i]
            if orig_clusters.get(cluster_idx) == None:
                orig_clusters[cluster_idx] = []
                proc_clusters[cluster_idx] = []
            (orig_clusters[cluster_idx]).append(original_comments[map_id[i]])
            (proc_clusters[cluster_idx]).append(proc_comments[i])

        with open(file_name, 'w', encoding="utf-8") as f:
            f.write("Number of clusters: {0:d} ({1:d} comments in total)\n\n".format(len(orig_clusters.keys()),len(list_clusters)))
            for key in orig_clusters.keys():
                f.write("CLUSTER %d (original %d comments):\n%s\n\n"%(key,len(orig_clusters[key]),orig_clusters[key]))
                f.write("CLUSTER %d (processed %d comments):\n%s\n\n\n"%(key,len(proc_clusters[key]),proc_clusters[key]))

    def save_topics(self, file_name: str, list_clusters: list):

        no_words = 0
        for key, topic in list_clusters:
                no_words += len(topic)

        with open(file_name, 'w', encoding="utf-8") as f:
            f.write("Number of topics: {0:d} ({1:d} words in total)\n\n".format(len(list_clusters),no_words))
            for key, topic in list_clusters:
                f.write("TOPIC %d:\n%s\n\n"%(key,topic))