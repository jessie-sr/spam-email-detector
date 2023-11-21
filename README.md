# Spam Email Detector (with Python)

## Introduction
This project implements a Spam Email Detector using Python, utilizing the K-Nearest Neighbours (KNN) algorithm. 

## K-Nearest Neighbours (KNN) algorithm Explained
KNN is a straightforward supervised learning algorithm that operates by finding the 'K' nearest neighbours of a test data point and classifying it based on the majority class among these neighbours. For email classification (spam or ham), the features compared are the frequencies of words in each email, using Euclidean distance as the measure of similarity.

## Data Set
The email data set is sourced from “The Enron-Spam datasets”, specifically the Enron2 dataset, containing 5857 emails labeled as either spam or ham.

## Libraries Used
```python
import os
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
```

## Implementation

### Loading and Preprocessing Data
Data loading involves reading email texts from files and storing them with labels. Preprocessing includes removing punctuation, converting to lowercase, and removing stopwords.

### Splitting Data
The dataset is divided into a training set (73%) and a testing set (27%).

### The KNN Algorithm
The KNN algorithm involves functions for counting word frequency, calculating Euclidean distance, selecting K nearest neighbours, and determining the class of a test email.

### Running the Classifier
The `main()` function orchestrates the process, calling functions for data loading, preprocessing, splitting, and running the KNN classifier.

### Output and Conclusion
With a K value of 7, the classifier achieves a 89.2% accuracy rate on a test data size of 1602 emails. However, it's noted that the process is time-intensive due to the high time complexity.

### How to Run the Program
To run the Spam Email Detector:

1. Ensure Python 3 and necessary libraries (numpy, sklearn, nltk) are installed on your system.
2. Download the Enron2 dataset from the [Enron-Spam datasets](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html) and extract it into a folder named `dataset` in the project directory.
3. Clone or download the Spam Email Detector code from the repository.
4. Open a terminal or command prompt and navigate to the project directory.
5. Run the program using the command `python spam_email_detector.py`, ensuring you have a function called `main()` in your script.
6. Optionally, modify the value of `K` in the `main()` function to test different numbers of nearest neighbours.

## Acknowledgements
This project utilizes the Enron-Spam dataset available at [Enron-Spam datasets](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html).
