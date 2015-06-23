__author__ = 'shirleyyoung'

# see more from the doc: https://radimrehurek.com/gensim/models/word2vec.html

from gensim.models import Word2Vec
modelName = "/path/Word2VectforNLPTraining"
model = Word2Vec.load(modelName)

# the model consists a feature vector for each word in the vocabulary, stored
# in a numpy array called "syn0"
print(type(model.syn0))

# number of words, number of features
print(model.syn0.shape)

# access individual word vector
# returns a 1 * # of features numpy array
print(model["man"])

# doesnt_match function tries to deduce which word in a set is most
# dissimilar from the others
print(model.doesnt_match("man woman child kitchen".split()) + '\n')
# not a good result
print(model.doesnt_match("paris berlin london austria".split()) + '\n')

# most_similar(): returns the score of the most similar words based on the criteria
# Find the top-N most similar words. Positive words contribute positively towards the
# similarity, negative words negatively.
print("most similar:")
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))
print()
print(model.most_similar("awful"))






