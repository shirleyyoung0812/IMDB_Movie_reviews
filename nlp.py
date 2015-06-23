__author__ = 'shirleyyoung'

import pandas as pd
from pysrc import processData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# quoting: int Controls whether quotes should be recognized
# 0, 1, 2, and 3 for
# QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONE, and QUOTE_NONNUMERIC, respectively
train = pd.read_csv("/path/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)



# clean the training data
num_reviews = train["review"].size
clean_train_reviews = []
print("Cleaning and parsing training data", end="\n")
for i in range(0, num_reviews):
    # if (i+1) % 1000 == 0:
    # print("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append(" ".join(processData.review_to_words(train["review"][i], True,False,False)))


# create features: convert a collection of text documents to a matrix of token counts
# i.e., get the frequency of each word
# max_features determine the maximum words that is taken into account, 5000 here
# e.g. dictionary {the, cat, sat, on, hat, dog, likes, and}
# sentence1: the cat sat on the hat {2, 1, 1, 1, 1, 0, 0, 0}
# sentence2: the dog likes the cat and the hat {3, 1, 0, 0, 1, 1, 1, 1}
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)


# learn the vocabulary dictionary and return term-document matrix
train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()

# use random forest to train the model
# sentiment is the label
# features are those that we have just created
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features,train["sentiment"])

# clean the test data
test = pd.read_csv("/path/testData.tsv",
                   header=0,
                   delimiter="\t", quoting=3)

num_test_reviews = len(test["review"])
clean_test_reviews = []
print("Cleaning and parsing testing data", end="\n")
for i in range(0, num_test_reviews):
    clean_test_reviews.append(" ".join(processData.review_to_words(test["review"][i], True)))

test_data_features = vectorizer.transform(clean_test_reviews).toarray()

# predict the sentiment
result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("BagOfWordsModel.csv", index=False, quoting=3)
