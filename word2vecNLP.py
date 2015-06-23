__author__ = 'shirleyyoung'

import pandas as pd
from pysrc import processData
import nltk.data
import logging
from gensim.models import word2vec

# train.shape: get the dimensions of the data set
train = pd.read_csv("/path/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("/path/unlabeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)

# use nltk to split the review to sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

bag_sentences = []
# Note we are appending lists of lists to bag_sentences.
# use bag_sentences.append()
# += join all the lists together
print("Parsing sentences from labeled training set")
for review in train["review"]:
    bag_sentences.append(processData.review_to_sentences(review, tokenizer, False, True, False))

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    bag_sentences.append(processData.review_to_sentences(review, tokenizer, False, True, False))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for the parameters in Word2Vec
num_features = 500  # word vector dimensionality
# minimum word count: any word that does not occur at least this many times
# across all documents is ignored
min_word_count = 40
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

print("Training model...")
model = word2vec.Word2Vec(bag_sentences, workers=num_workers,
                          size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient
model.init_sims(replace=True)

# save the model for later use
# call Word2Vec.load()


model.save("Word2VectforNLPTraining")



