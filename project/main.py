import json
from compare_clustering_solutions import evaluate_clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import spacy
from sklearn.cluster import KMeans
import warnings
from sklearn.decomposition import LatentDirichletAllocation

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
stopwords = list(nlp.Defaults.stop_words)
# filter out the specific RuntimeWarning
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I had",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

############################## data reading ##############################
def encode_requests(df):
    embeddings = model.encode(df.iloc[:, 1])
    return embeddings


########################### Clustering functions ###########################
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
    similarity_threshold = 0.87
    cluster_assignments = [-1 for _ in range(n_requests)]
    clusters = []

    for request_index, point in enumerate(embeddings):
        # point = point.todense()
        min_distance, closest_cluster = get_min_distance_and_closest_cluster(clusters, point)
        if closest_cluster is not None and min_distance < similarity_threshold:
            insert_to_cluster(clusters, cluster_assignments, request_index, closest_cluster, point)
        else:
            n_clusters += 1
            create_new_cluster(clusters, cluster_assignments, request_index, n_clusters, point)
    
    # Filter out clusters with less than min_size requests
    generated_clusters, non_generated_clusters = filter_clusters(df, clusters, min_size)

    return generated_clusters, non_generated_clusters


########################### representatives ###########################
def find_representatives(generated_clusters, num_rep):
    for cluster in generated_clusters:
        requests = cluster['requests']
        vectors = model.encode(requests)

        # Find the centroid of all the vectors
        centroid = np.mean(vectors, axis=0)

        # Apply KMeans++ algorithm to vectors
        kmeans = KMeans(n_clusters=num_rep, init='k-means++')
        kmeans.fit(vectors)
        
        # Find the most diverse vector in each cluster
        representatives = []
        for i in range(num_rep):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_vectors = vectors[cluster_indices]
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            furthest_index = np.argmax(distances)
            representatives.append(requests[cluster_indices[furthest_index]])

        # Map the num_rep data points back to the original sentences'
        cluster['representative_sentences'] = representatives



########################### Labeling clusters ###########################

def copy_clusters_list(clusters):
    new_list = []
    for item in clusters:
        new_list.append(dict(item))
    return new_list
        
def remove_stopwords(clusters):
    res_clusters = copy_clusters_list(clusters)
    
    # initiating empty list
    for i in range(len(res_clusters)):
        res_clusters[i].pop("requests")
        res_clusters[i]["requests"] = []
    
    for i, cluster in enumerate(clusters):
        
        for request in cluster["requests"]:
            r = ""
            for word in request.split():                
                if word not in stopwords:
                    r += word + " "
            if len(r) != 0:        
                res_clusters[i]["requests"].append(r)
            else :
                res_clusters[i]["requests"].append(request)                
    
    return  res_clusters

def text_processing(clusters):
    '''
    in this function we will remove stopwords and contractions in order 
    to get a better result in the labeling process
    '''
    res_clusters = copy_clusters_list(clusters)
    
    # initiating empty list
    for i in range(len(res_clusters)):
        res_clusters[i].pop("requests")
        res_clusters[i]["requests"] = []
    
    for i, cluster in enumerate(clusters):
        
        for request in cluster["requests"]:
            r = ""
            for word in request.split():                
                if word in contractions.keys():
                   r += contractions[word] + " "
                else:
                    r += word + " "
            res_clusters[i]["requests"].append(r)
            
    return remove_stopwords(res_clusters)       
                    
def clusters_labeling(generated_clusters): 
    processed_clusters = text_processing(generated_clusters)
    clusters_names = []
    
    for i, cluster in enumerate(processed_clusters):        
        # Initialize the TfidfVectorizer
        tfidf = TfidfVectorizer(ngram_range= (1, 3))        
        # Fit and transform the clusters data
        tfidf_matrix = tfidf.fit_transform(cluster["requests"])
        
        # create an LDA object
        lda = LatentDirichletAllocation(n_components=2, random_state=42)
        # fit the LDA model to the TF-IDF matrix
        lda.fit(tfidf_matrix)
        
        # find the topics of the clusters
        words = tfidf.get_feature_names_out()
        topics = [[words[i] for i in topic.argsort()[:-2 - 1:-1]] for (topic_idx, topic) in enumerate(lda.components_)]
        topics = np.array(topics).ravel()
        
        ngrams = list(words)
        ngram_counts = np.zeros(len(ngrams))
        topics = list(topics)
        
        
        seperated = []
        for t in topics:
            seperated += t.split()
        
        # Count the number of times each topic-word appears in the n-gram
        for n in ngrams:
            for word in n.split():
                if word in seperated:
                    ngram_counts[ngrams.index(n)] += 1
        
        # select the n-gram with the highest count to be the cluster name
        max_idx = ngram_counts.argmax()
        generated_clusters[i]["cluster_name"] = ngrams[max_idx]
        clusters_names.append(ngrams[max_idx])
                       
        
            
########################### Writing to Output file ###########################
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
    clusters_labeling(generated_clusters)
    # Save results to output file
    output_data(output_file, generated_clusters, non_generated_clusters)

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
