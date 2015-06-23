"""
This script defines function used to pre-process data before training

Word2Vec expects single sentences, each one as a list of words.
This module defines a function to fulfill this requirement
use nltk.tokenize.punkt module to split a text into a list of sentences,
by using an unsupervised algorithm to build a model for abbreviation words,
collocations and words that start sentences.
It must be trained on a large collection of plaintext in the target language before
it can be used.
see doc: http://www.nltk.org/api/nltk.tokenize.html
"""
__author__ = 'shirleyyoung'

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re

def review_to_words(raw_review, remove_stopwords=False, remove_numbers=False, remove_smileys=False):
    # use BeautifulSoup library to remove the HTML/XML tags (e.g., <br />)
    review_text = BeautifulSoup(raw_review).get_text()

    # emotional symbols may affect the meaning of the review
    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
                :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))

    # [^] matches a single character that is not contained within the brackets
    # re.sub() replace the pattern by the desired character/string
    if remove_numbers and remove_smileys:
        # any character that is not in a to z and A to Z (non text)
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
    elif remove_smileys:
         # numbers are also included
        review_text = re.sub("[^a-zA-Z0-9]", " ", review_text)
    elif remove_numbers:
        review_text = re.sub("[^a-zA-Z" + smiley_pattern + "]", " ", review_text)
    else:
        review_text = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", review_text)


    # split in to a list of words
    words = review_text.lower().split()

    if remove_stopwords:
        # create a set of all stop words
        stops = set(stopwords.words("english"))
        # remove stop words from the list
        words = [w for w in words if w not in stops]

    # for bag of words, return a string that is the concatenation of all the meaningful words
    # for word2Vector, return list of words
    # return " ".join(words)
    return words

def review_to_sentences(review, tokenizer, remove_stopwords=False, remove_numbers=False, remove_smileys=False):
    """
    This function splits a review into parsed sentences
    :param review:
    :param tokenizer:
    :param remove_stopwords:
    :return: sentences, list of lists
    """
    # review.strip()remove the white spaces in the review
    # use tokenizer to separate review to sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    #cleaned_review = [review_to_words(sentence, remove_stopwords, remove_numbers, remove_smileys) for sentence
    #                  in raw_sentences if len(sentence) > 0]
    # generic form equals append
    cleaned_review = []
    for sentence in raw_sentences:
        if len(sentence) > 0:
            cleaned_review += review_to_words(sentence, remove_stopwords, remove_numbers, remove_smileys)

    return cleaned_review

def clean_data(data):
    """
    clean the training and test data and return a list of words
    :param data:
    :return:
    """
    # raise an error if there is no review column
    try:
        reviewsSet = data["review"]
    except ValueError:
        print('No "review" column!')
        raise

    cleaned_data = [review_to_words(review, True, True, False) for review in reviewsSet]
    # for review in reviewsSet:
    #  cleaned_data.append(review_to_words(review, True, True, False))
    return cleaned_data
