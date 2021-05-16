import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

from SpamFilterFromScratch import LogisticRegression
#from regression import LogisticRegression


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


#bc = datasets.load_breast_cancer()
Spam_Filter = pd.read_csv('emails.csv')
print(Spam_Filter.drop_duplicates(inplace=True))


def process_text(text):
    # 1 remove punctuation
    # 2 remove stopwords
    # 3 return a list of clean text words

    # 1
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    # 2
    clean_words = [word for word in no_punctuation.split(
    ) if word.lower() not in stopwords.words('english')]
    # 3
    return clean_words


# Show the tokenization ( a list of tokens also called lemmas)
print(Spam_Filter['text'].head().apply(process_text))

# Convert a collection of text to a matrix of tokens
messages_bow = CountVectorizer(
    analyzer=process_text).fit_transform(Spam_Filter['text'])

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    messages_bow, Spam_Filter['spam'], test_size=0.20, random_state=0)

# Get the shape of the messages_bow
print(messages_bow.shape)

#X, y = bc.data, bc.target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1234)

regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("LR classification accuracy:", accuracy(y_test, predictions))
