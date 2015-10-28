__author__ = 'nisarg'

import string
import json

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Function to break text into "tokens", lowercase them, remove punctuation and stopwords, and stem them
def tokenize(text):
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]


import os
os.environ["SPARK_HOME"] = "/home/nisarg/spark-1.5.0-bin-hadoop2.6"


# Initialize a SparkContext
sc = SparkContext()

# Import full dataset of newsgroup posts as text file
data_raw = sc.textFile('./20_newsgroups/alt.atheism/')

# Parse JSON entries in dataset
data = data_raw.map(lambda line: json.loads(json.dumps(line)))

# Extract relevant fields in dataset -- category label and text content
data_pared = data.map(lambda line: (line['label'], line['text']))

# Prepare text for analysis using our tokenize function to clean it up
data_cleaned = data_pared.map(lambda (label, text): (label, tokenize(text)))

# Hashing term frequency vectorizer with 50k features
htf = HashingTF(50000)

# Create an RDD of LabeledPoints using category labels as labels and tokenized, hashed text as feature vectors
data_hashed = data_cleaned.map(lambda (label, text): LabeledPoint(label, htf.transform(text)))

# Ask Spark to persist the RDD so it won't have to be re-created later
data_hashed.persist()

# Split data 70/30 into training and test data sets
train_hashed, test_hashed = data_hashed.randomSplit([0.7, 0.3])

# Train a Naive Bayes model on the training data
model = NaiveBayes.train(train_hashed)

# Compare predicted labels to actual labels
prediction_and_labels = test_hashed.map(lambda point: (model.predict(point.features), point.label))

# Filter to only correct predictions
correct = prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)

# Calculate and print accuracy rate
accuracy = correct.count() / float(test_hashed.count())

print "Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"

