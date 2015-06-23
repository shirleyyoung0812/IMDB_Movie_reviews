"""
(almost)Every review is in different length, thus we need to find a way to take
individual word vectors and transform into a feature set that is the same length for
every review.
This script averages the word vectors in a given review.
i.e., add the feature vector of each word in the review together then average the
sum feature vector
In order to remove noise, it is better to remove stop words.
The predictions are made by using random forest (again)
"""
__author__ = 'shirleyyoung'

import numpy as np
import pandas as pd
from pysrc import processData
from gensim.models import Word2Vec
import sys
sys.path.insert(0, '/path/')
import randomForestClassifier






def makeFeatureVec(review, model, num_features):
    """
    given a review, define the feature vector by averaging the feature vectors
    of all words that exist in the model vocabulary in the review
    :param review:
    :param model:
    :param num_features:
    :return:
    """

    featureVec = np.zeros(num_features, dtype=np.float32)
    nwords = 0

    # index2word is the list of the names of the words in the model's vocabulary.
    # convert it to set for speed
    vocabulary_set = set(model.index2word)

    # loop over each word in the review and add its feature vector to the total
    # if the word is in the model's vocabulary
    for word in review:
        if word in vocabulary_set:
            nwords = nwords + 1
            # add arguments element-wise
            # if x1.shape != x2.shape, they must be able to be casted
            # to a common shape
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs (reviewSet, model):

    # initialize variables
    counter = 0
    num_features = model.syn0.shape[1]
    reviewsetFV = np.zeros((len(reviewSet),num_features), dtype=np.float32)

    for review in reviewSet:
        reviewsetFV[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewsetFV

def main():
    """
    main function to make prediction
    use random forest
    :return:
    """
    train = pd.read_csv("/path/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("/path/testData.tsv",
                   header=0, delimiter="\t", quoting=3)

    modelName = "/path/Word2VectforNLPTraining"
    model = Word2Vec.load(modelName)

    print("Processing training data...")
    cleaned_training_data = processData.clean_data(train)
    trainingDataFV = getAvgFeatureVecs(cleaned_training_data,model)
    print("Processing test data...")
    cleaned_test_data = processData.clean_data(test)
    testDataFV = getAvgFeatureVecs(cleaned_test_data,model)

    n_estimators = 100
    result = randomForestClassifier.rfClassifer(n_estimators, trainingDataFV, train["sentiment"],testDataFV)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AvgVecPredict.csv", index=False, quoting=3)


if __name__ == '__main__':
    main()

