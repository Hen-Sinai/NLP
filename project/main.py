import json
from compare_clustering_solutions import evaluate_clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def encode_requests(df):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(df.iloc[:, 1])
    return embeddings

def get_min_distance_and_closest_cluster(clusters, point):
    min_distance = float("inf")
    closest_cluster = None
    for cluster_index, cluster in enumerate(clusters):
        #  it calculates the euclidean distance between the given request feature and the centroid of the cluster.
        distance = pairwise_distances(np.asarray(point).reshape(1,-1), np.asarray(cluster['centroid']).reshape(1,-1))
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster_index
    
    return min_distance, closest_cluster

def insert_to_cluster(clusters, cluster_assignments, request_index, closest_cluster, point):
    cluster_assignments[request_index] = closest_cluster
    clusters[closest_cluster]['requests'].append(request_index)
    clusters[closest_cluster]['centroid'] = (clusters[closest_cluster]['centroid'] * clusters[closest_cluster]['size']+point) / (clusters[closest_cluster]['size'] + 1)
    clusters[closest_cluster]['size'] += 1

def create_new_cluster(clusters, cluster_assignments, request_index, n_clusters, point):
    cluster_assignments[request_index] = n_clusters
    clusters.append({'requests': [request_index], 'centroid': point, 'size': 1})

def filter_clusters(df, clusters, min_size):
    generated_clusters = []
    non_generated_clusters = []
    for cluster in clusters:
        if cluster['size'] >= int(min_size):
            generated_clusters.append({'requests': [df.loc[i][1] for i in cluster['requests']]})
        else:
            non_generated_clusters = non_generated_clusters + [df.loc[i][1] for i in cluster['requests']]
    
    return generated_clusters, non_generated_clusters

def perform_clustering(df, embeddings, min_size):    
    # Initialize variables
    n_requests = len(df)
    n_clusters = 0
    similarity_threshold = 1.06
    cluster_assignments = [-1 for _ in range(n_requests)]
    clusters = []

    for request_index, point in enumerate(embeddings):
        point = point.todense()
        min_distance, closest_cluster = get_min_distance_and_closest_cluster(clusters, point)
        if closest_cluster is not None and min_distance < similarity_threshold:
            insert_to_cluster(clusters, cluster_assignments, request_index, closest_cluster, point)
        else:
            n_clusters += 1
            create_new_cluster(clusters, cluster_assignments, request_index, n_clusters, point)
    
    # Filter out clusters with less than min_size requests
    generated_clusters, non_generated_clusters = filter_clusters(df, clusters, min_size)
    print(len(generated_clusters))
    return generated_clusters, non_generated_clusters


def get_rep_for_cluster(requests, num_rep, vectorizer):
    # Fit the requests
    tfidf_matrix = vectorizer.fit_transform(requests)

    # Convert the TF-IDF matrix to a NumPy array
    tfidf_array = np.asarray(tfidf_matrix.todense())

    # Calculate the centroid of the tfidf matrix
    centroid = np.mean(tfidf_array, axis=0)
    centroid = np.reshape(centroid, (1, -1))

    # Calculate cosine distances between each requests and the centroid
    distances = cosine_distances(tfidf_array, centroid)

    # Get the indices of the 3 farthest requests from the centroid
    farthest_indices = np.argsort(distances, axis=0)[-num_rep:].ravel()

    # Append the farthest sentences to the list
    farthest_sentences = []
    for index in farthest_indices:
        farthest_sentences.append(requests[index])

    return farthest_sentences

def find_representatives(generated_clusters, num_rep):
    vectorizer = TfidfVectorizer()
    for cluster in generated_clusters:
        cluster['representative_sentences'] = get_rep_for_cluster(cluster['requests'], num_rep, vectorizer)

def get_title(requests):
    doc = nlp(" ".join(requests))
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(words[:2])

def name_clusters(generated_clusters):
    for cluster in generated_clusters:
        cluster['cluster_name'] = get_title(cluster['requests'])



def output_data(output_file, generated_clusters, non_generated_clusters):
    with open(output_file, 'w') as f:
        json.dump({'cluster_list': generated_clusters, 'unclustered': non_generated_clusters}, f, indent=4)

def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    # Read data from file
    df = pd.read_csv(data_file, sep=',')
    embeddings = encode_requests(df)
    # Perform clustering
    generated_clusters, non_generated_clusters = perform_clustering(df, embeddings, min_size)
    # Find representatives
    find_representatives(generated_clusters, int(num_rep))
    # Name clusters
    name_clusters(generated_clusters)
    # Save results to output file
    output_data(output_file, generated_clusters, non_generated_clusters)
    
    # print(len(generated_clusters))


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    # evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    evaluate_clustering(config['example_solution_file'], config['output_file'])