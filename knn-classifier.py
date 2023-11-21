import os
import string
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
Implementation:
1. Load the spam and ham emails
2. Remove common punctuation and symbols
3. Lowercase all letters
4. Remove stopwords (very common words like pronouns, articles, etc.)
5. Split emails into training email and testing emails
6. For each test email, calculate the similarity between it and all training emails
    6.1. For each word that exists in either test email or training email, count its frequency in both emails
    6.2. calculate the euclidean distance between both emails to determine similarity
7. Sort the emails in ascending order of euclidean distance
8. Select the k nearest neighbors (shortest distance)
9. Assign the class which is most frequent in the selected k nearest neighbours to the new email
'''

def load_data():
    print("Loading data...")
    ham_files = os.listdir("dataset/ham")
    spam_files = os.listdir("dataset/spam")
    data = []

    for file_path in ham_files:
        with open(f"dataset/ham/{file_path}", "r", encoding='latin1') as f:  # Note the encoding change here
            data.append([f.read(), "ham"])

    for file_path in spam_files:
        with open(f"dataset/spam/{file_path}", "r", encoding='latin1') as f:  # And here
            data.append([f.read(), "spam"])

    return np.array(data)

def preprocess_data(data):
    print("Preprocessing data...")
    punc = string.punctuation
    sw = stopwords.words('english')

    for record in data:
        record[0] = record[0].translate(str.maketrans('', '', punc)).lower()
        record[0] = " ".join([word for word in record[0].split() if word not in sw])
    
    return data

def split_data(data):
    print("Splitting data...")
    features, labels = data[:, 0], data[:, 1]
    return train_test_split(features, labels, test_size=0.27, random_state=42)

# Helper functions.
def get_count(text):
    wordCounts = {}
    for word in text.split():
        wordCounts[word] = wordCounts.get(word, 0) + 1
    return wordCounts

def euclidean_difference(test_counts, training_counts):
    total = sum((test_counts.get(word, 0) - training_counts.get(word, 0))**2 
                for word in set(test_counts.keys()).union(training_counts.keys()))
    return total**0.5

def get_class(k_values):
    counts = {'spam': 0, 'ham': 0}
    for label, _ in k_values:
        counts[label] += 1
    return 'spam' if counts['spam'] > counts['ham'] else 'ham'

# KNN Classifier
def knn_classifier(training_data, training_labels, test_data, k):
    print("Running KNN Classifier...")
    results = []
    training_word_counts = [get_count(text) for text in training_data]

    for test_text in test_data:
        test_word_counts = get_count(test_text)
        distances = []

        for i, train_counts in enumerate(training_word_counts):
            euclidean_diff = euclidean_difference(test_word_counts, train_counts)
            distances.append((training_labels[i], euclidean_diff))

        distances.sort(key=lambda x: x[1])
        k_nearest = distances[:k]
        results.append(get_class(k_nearest))

    return results

def main(k=5):
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)

    predicted_labels = knn_classifier(training_data, training_labels, test_data, k)
    accuracy = accuracy_score(test_labels, predicted_labels)

    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main(11)  # You can change the value of K here


