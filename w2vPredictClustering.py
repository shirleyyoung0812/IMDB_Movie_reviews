__author__ = 'shirleyyoung'

import sys
sys.path.insert(0, '/path/')
import kMeans
import randomForestClassifier
import time
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from pysrc import processData

#def create_bag_of_centroids(review, index_word_map, num_centroids):
#    """
#    assign each word in the review to a centroid
#    this returns a numpy array with the dimension as num_clusters
#    each will be served as one feature for classification
#    :param review:
#    :param index_word_map:
#    :return:
#    """
#    featureVector = np.zeros(num_centroids, dtype=np.float)
#    for word in review:
#        if word in index_word_map:
#            index = index_word_map[word]
#            featureVector[index] += 1
#    return featureVector


def main():

    modelName = "Word2VectforNLPTraining"
    model = Word2Vec.load(modelName)

    # model.init_sims(replace=True)

    word_vectors = model.syn0
    # print(word_vectors[0])
    num_clusters = int(word_vectors.shape[0] / 5)
    # print("number of clusters: {}".format(num_clusters))
    # input("Press enter to continue:")
    print("Clustering...")
    startTime = time.time()
    cluster_index = kMeans.kmeans(num_clusters, word_vectors)
    endTime = time.time()

    print("Time taken for clustering: {} seconds".format(endTime - startTime))


    # create a word/index dictionary, mapping each vocabulary word to a cluster number
    # zip(): make an iterator that aggregates elements from each of the iterables
    index_word_map = dict(zip(model.index2word, cluster_index))

    def create_bag_of_centroids(reviewData):
        """
        assign each word in the review to a centroid
        this returns a numpy array with the dimension as num_clusters
        each will be served as one feature for classification
        :param reviewData:
        :return:
        """
        featureVector = np.zeros(num_clusters, dtype=np.float)
        for word in reviewData:
            if word in index_word_map:
                index = index_word_map[word]
                featureVector[index] += 1
        return featureVector

    train = pd.read_csv("/path/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("/path/testData.tsv",
                   header=0, delimiter="\t", quoting=3)

    trainingDataFV = np.zeros((train["review"].size, num_clusters), dtype=np.float)
    testDataFV = np.zeros((test["review"].size, num_clusters), dtype=np.float)

    print("Processing training data...")
    counter = 0
    cleaned_training_data = processData.clean_data(train)
    for review in cleaned_training_data:
        trainingDataFV[counter] = create_bag_of_centroids(review)
        counter += 1

    print("Processing test data...")
    counter = 0
    cleaned_test_data = processData.clean_data(test)
    for review in cleaned_test_data:
        testDataFV[counter] = create_bag_of_centroids(review)
        counter += 1

    n_estimators = 100
    result = randomForestClassifier.rfClassifer(n_estimators, trainingDataFV, train["sentiment"],testDataFV)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Doc2Vec_Clustering.csv", index=False, quoting=3)

if __name__ == '__main__':
    main()