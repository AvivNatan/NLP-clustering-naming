import json
import time
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from project.compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer

similarity_threshold = 0.8
max_iterations = 50
centroids_change_threshold = 70


def analyze_unrecognized_requests(data_file, output_file, min_size):
    start = time.time()

    # Step 1: Initialize BERT Model, Get requests from file
    MODEL_NAME = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(MODEL_NAME)
    id_to_requests = read_requests_from_file(data_file)

    # Step 2: Encode Sentences and normalize
    embeddings = model.encode(list(id_to_requests.values()))
    normalized_embeddings = get_normalized_embeddings(embeddings)
    id_to_embeddings = {id: emb for (id, _), emb in zip(id_to_requests.items(), normalized_embeddings)}

    # Step 3: Clustering
    clusters = get_clusters(id_to_embeddings)

    # Step 4: Extract requests
    clusters, unclustered = get_requests_for_clusters(clusters, id_to_requests, int(min_size))

    # Step 5: Name the clusters
    clusters_names = label_clusters(clusters, model)

    # Step 6: Write to file
    clusters_list = get_final_clustering_with_labeling(clusters, unclustered, clusters_names)
    with open(output_file, 'w') as file:
        json.dump(clusters_list, file, indent=4)

    print(f'total time: {round(time.time() - start, 0)} sec')


# function build dictionary of final result to write it to the file.
def get_final_clustering_with_labeling(clusters, unclustered, clusters_names):
    clustering = list()
    for index_cluster, cluster in enumerate(clusters):
        cluster_dict = {'cluster_name': clusters_names[index_cluster], 'requests': cluster}
        clustering.append(cluster_dict)

    result = {'cluster_list': clustering, 'unclustered': unclustered}
    return result


# extract the requests according to their ids
def get_requests_for_clusters(clusters, id_to_requests, min_size):
    res = list()
    unclustered = list()
    for cluster in clusters:
        list_cluster = list(cluster)
        if len(cluster) >= min_size:
            requests = [id_to_requests[id] for id in list_cluster]
            res.append(requests)
        else:
            for id in list_cluster:
                unclustered.append(id_to_requests[id])

    return res, unclustered


# function to read requests from the file
# function build dictionary thst his keys are ids and values are the requests
def read_requests_from_file(data_file):
    id_to_requests = dict()
    with open(data_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            id_to_requests[row['id']] = row['text'].lower().strip()
    return id_to_requests


def get_normalized_embeddings(embeddings):
    # Compute the L2 norm of each embedding vector
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Normalize the embeddings
    return embeddings / embedding_norms


# function check similarity
def check_similarity(emb, clusters_centroid):
    # check if there is any clusters for similarity check
    if not clusters_centroid:
        return False, -1
    index_cluster, similarity = get_max_similarity(emb, clusters_centroid)
    # min distance need to be smaller then threshold
    return similarity <= similarity_threshold, index_cluster


# calc the highest similarity from the embeddings_to_check(centroids) to embedding.
def get_max_similarity(embedding, embeddings_to_check):
    # calc the similarity using Euclidean distances.
    distances = np.linalg.norm(np.array(embeddings_to_check) - embedding, axis=1)

    # Find the index of the emb(centroid) with the highest similarity(lowest distance)
    closest_index = np.argmin(distances)

    # return the index
    # return the highest similarity to embedding
    return closest_index, distances[closest_index]


# function calc the cluster centroid
def calc_centroid(old_centroid, cluster, id_to_embeddings):
    # check if the cluster is out of requests - empty
    if not cluster:
        return np.zeros_like(old_centroid)
    # takes all the embeddings of the cluster and calc their mean - the centroid
    embeddings = [id_to_embeddings[id] for id in list(cluster)]
    centroid = np.mean(embeddings, axis=0)
    return centroid


# update all centroids according to ids clusters that was changed
def update_centroids(clusters_centroid, clusters, id_to_embeddings, cluster_ids_centroid_change):
    # update the centroids
    for i in list(cluster_ids_centroid_change):
        clusters_centroid[i] = calc_centroid(clusters_centroid[i], clusters[i], id_to_embeddings)
    cluster_ids_centroid_change = set()
    return clusters_centroid, cluster_ids_centroid_change


def assign_and_remove_request(id_to_cluster, clusters, id, index_new_cluster, cluster_ids_centroid_change):
    index_old_cluster = id_to_cluster.get(id)
    # assign the cluster to the id
    id_to_cluster[id] = index_new_cluster
    if index_old_cluster is None:
        return
    cluster_ids_centroid_change.add(index_old_cluster)
    # remove the id of the request from cluster
    clusters[index_old_cluster].remove(id)


def get_clusters(id_to_embeddings):

    # list of clusters - each cluster is a set of ids (of requests)
    clusters = list()
    # list of clusters centroid - the centroid of clusters[i] is in clusters_centroid[i]
    clusters_centroid = list()
    # dictionary id to its cluster index.
    id_to_cluster = dict()
    # set of clusters index that have changed and need to update their centroid
    ids_centroids_changed = set()

    for _ in range(max_iterations):
        # to know if any cluster was updated
        updated = False
        for id, emb in id_to_embeddings.items():

            # update the centroids if I got to threshold
            if len(ids_centroids_changed) == centroids_change_threshold:
                clusters_centroid, ids_centroids_changed = update_centroids(clusters_centroid,
                                            clusters, id_to_embeddings, ids_centroids_changed)

            is_similarity, index_cluster = check_similarity(emb, clusters_centroid)
            if is_similarity:
                is_id_in_cluster = id_to_cluster.get(id) == index_cluster
                # need to update the cluster of the request
                if not is_id_in_cluster:
                    clusters[index_cluster].add(id)
                    ids_centroids_changed.add(index_cluster)
                    assign_and_remove_request(id_to_cluster, clusters, id, index_cluster, ids_centroids_changed)
                    updated = True
            # request doesn't feet to any of the clusters
            else:
                # open new cluster with himself
                clusters.append({id})
                assign_and_remove_request(id_to_cluster, clusters, id, len(clusters) - 1, ids_centroids_changed)
                # the cluster centroid will be the embedding of the one request
                clusters_centroid.append(emb)
                updated = True

        clusters_centroid, ids_centroids_changed = update_centroids(clusters_centroid,
                                                clusters, id_to_embeddings, ids_centroids_changed)

        # check convergence
        if not updated:
            break

    return clusters


# function find names for clusters
# for each cluster function find the ngram most similar to a combine document of cluster requests
def label_clusters(clusters, model):
    clusters_labels = list()
    clusters_top20ngrams = get_top_ngram_for_clusters(clusters)
    for i, requests in enumerate(clusters):
        # Combine sentences into a single "document" for the cluster
        cluster_text = " ".join(req for req in requests)
        # Extract document embedding
        document_embedding = model.encode(cluster_text)
        document_embedding_normalized = get_normalized_embeddings([document_embedding])
        # Extract ngrams embedding
        ngrams_embeddings = model.encode(clusters_top20ngrams[i])
        ngrams_embeddings_normalized = get_normalized_embeddings(ngrams_embeddings)
        # Calculate similarity between ngrams and document embedding
        index_ngram, similarity = get_max_similarity(document_embedding_normalized, ngrams_embeddings_normalized)
        cluster_name = clusters_top20ngrams[i][index_ngram]
        clusters_labels.append(cluster_name)

    return clusters_labels


# Function get the 20 ngram with top TF-IDF score
# Each ngram is 2-5 words
def get_top_ngram_for_clusters(clusters):
    clusters_top20ngrams = list()

    for cluster_requests in clusters:
        # Calculate tf-idf for n-grams
        vectorizer = TfidfVectorizer(ngram_range=(2, 5))
        tfidf_matrix = vectorizer.fit_transform(cluster_requests)
        feature_names = vectorizer.get_feature_names_out()
        # Find the 20 n-gram with the highest TF-IDF score
        top_tfidf_indices = tfidf_matrix.max(axis=0).toarray().flatten().argsort()[-20:][::-1]
        top_ngrams = [feature_names[i] for i in top_tfidf_indices]
        clusters_top20ngrams.append(top_ngrams)

    return clusters_top20ngrams


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])


