#!/usr/python3
# Clustering of web pages using 'bag of words'
# and TF-IDF representation
# ---------------------------------------------

# Resources and links:
# * NLTK: http://www.nltk.org/
# * Beautiful Soup: http://www.crummy.com/software/BeautifulSoup/
# * Scikit-learn: http://scikit-learn.org/
# * Recipe: Text clustering using NLTK and scikit-learn: https://nlpforhackers.io/recipe-text-clustering/
# * Document Similarity using NLTK and Scikit-Learn: http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
# * Representación de documentos mediante TF-IDF: https://www.youtube.com/watch?v=OkSZZ0F7ToA

# Previously:
# * Install NLTK
#   python3 -m pip install --user -U nltk
# * Install beautifulsoup4
#   python3 -m pip install --user -U beautifulsoup4
# * Install scikit-learn
#   python3 -m pip install --user -U scikit-learn
# * Download html files in 'path' directory (see 'descargas.scr')

import nltk
import string
import os
import collections

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Read all html files in path and transform in text
#   If you get an UnicodeDecodeError remove file content in 'file' varaible
path = './html'
translate_table = dict((ord(char), ' ') for char in string.punctuation)
token_dict = {}
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        html = shakes.read()
        # Extract text from html
        text = BeautifulSoup(html).get_text().encode('ascii', 'ignore')
        # Lowercase and remove punctuation
        lowers = str(text.lower())
        no_punctuation = lowers.translate(translate_table)
        token_dict[file] = no_punctuation

# Show number of reading files
len(token_dict)

# Tokenizer function
def process_text(text, stem=True):
    """Tokenize text and stem words removing punctuation"""
    # text = text.translate(None, string.punctuation)
    tokens = word_tokenize(text)
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# Transform texts to Tf-Idf coordinates
# (http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
#  If you get an "UserWarning: Your stop_words may be inconsistent" ignore it
stop_words = [process_text(w)[0] for w in stopwords.words('english')]
vectorizer = TfidfVectorizer(tokenizer=process_text,
                             stop_words=stop_words,
                             max_df=0.5,
                             min_df=0.1,
                             lowercase=True)
tfidf_model = vectorizer.fit_transform(token_dict.values())

# Show dimensions of tfidf_model
tfidf_model

# Show selected tokens
#   On old versions of scikit-learn change 'get_feature_names_out' method by
#   'get_feature_names'
feature_names = vectorizer.get_feature_names_out()
feature_names

# Example of a sentence Tf-Idf vectorization
sentence = 'this sentence has seen text such as computer but also animals films or kings'
response = vectorizer.transform([sentence])
for col in response.nonzero()[1]:
    print(feature_names[col], ' - ', response[0, col])

# Cluster texts using K-Means
# (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
#   On old versions of scikit-learn remove 'n_init' parameter
km_model = KMeans(n_clusters=3, n_init='auto', verbose=0)
km_model.fit(tfidf_model)

# Print clusters
clusters = collections.defaultdict(list)
for idx, label in enumerate(km_model.labels_):
    clusters[label].append(idx)

dict(clusters)

# Print number of elements of each cluster
for key, elements in dict(clusters).items():
    print(str(key) + ':', len(elements))

# Print labels of each url webpage
key = list(token_dict.keys())
for idx, label in enumerate(km_model.labels_):
    print(str(label) + ':', key[idx].replace('_','/').replace('.html',''))

# Print 4 most relevant tokens of each cluster
kmcc = km_model.cluster_centers_.copy()
for idx, item in enumerate(dict(clusters).items()):
    print(str(item[0]) + ':')
    for j in range(4):
        idxmax = kmcc[idx].argmax()
        print('  ', feature_names[idxmax], ' - ', kmcc[idx][idxmax])
        kmcc[idx][idxmax] = 0.0
    print()