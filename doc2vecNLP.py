__author__ = 'shirleyyoung'

from pysrc import processData
from gensim.models import doc2vec
import pandas as pd
import numpy as np
import nltk.data
import logging

train = pd.read_csv("/path/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
# test = pd.read_csv("/Users/shirleyyoung/Documents/Kaggle/Bag_of_Words_Meets_Bags_of_Popcorn/testData.tsv",
#                   header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("/path/unlabeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
labeled = [processData.review_to_sentences(review, tokenizer) for review in train["review"]]
unlabeled = [processData.review_to_sentences(review, tokenizer) for review in unlabeled_train["review"]]

# print(type(labeled[0]))
# print(labeled[0])
# input("Press enter to continue...")

def labelizeReviews(reviewSet, labelType):
    """
    add label to each review
    :param reviewSet:
    :param label: the label to be put on the review
    :return:
    """
    labelized = []
    for index, review in enumerate(reviewSet):

        labelized.append(doc2vec.LabeledSentence(words=review, labels=['%s_%s'%(labelType, index)]))
    return labelized
# the input to doc2vec is an iterator of LabeledSentence objects
# each consists a list of words and alist of labels
labeled = labelizeReviews(labeled, 'LABELED')
unlabeled = labelizeReviews(unlabeled, 'UNLABELED')



bag_labeled_sentence = labeled + unlabeled


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
# print(type(bag_labeled_sentence))
# for i in range(3):
#    print(str(bag_labeled_sentence[i]))
#    print()
# input("Press enter to continue...")
# parameter values
num_features = 500
# minimum word count: any word that does not occur at least this many times
# across all documents is ignored
min_word_count = 40
# the paper (http://arxiv.org/pdf/1405.4053v2.pdf) suggests 10 is the optimal
context = 10
#  threshold for configuring which higher-frequency words are randomly downsampled;
# default is 0 (off), useful value is 1e-5
# set the same as word2vec
downsampling = 1e-3
um_workers = 4  # Number of threads to run in parallel

# if sentence is not supplied, the model is left uninitialized
# otherwise the model is trained automatically
# https://www.codatlas.com/github.com/piskvorky/gensim/develop/gensim/models/doc2vec.py?line=192
model = doc2vec.Doc2Vec(size=num_features,
                        window=context, min_count=min_word_count,
                        sample=downsampling, workers=4)

model.build_vocab(bag_labeled_sentence)
# gensim documentation suggests training over data set for multiple times
# by either randomizing the order of the data set or adjusting learning rate
# see here for adjusting learn rate: http://rare-technologies.com/doc2vec-tutorial/
# iterate 10 times
for it in range(10):
    # perm = np.random.permutation(bag_labeled_sentence.shape[0])
    model.train(np.random.permutation(bag_labeled_sentence))

# model.init_sims(replace=True)
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))
model.save("Doc2VectforNLPTraining")