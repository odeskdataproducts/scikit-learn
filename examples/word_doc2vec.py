"""
==============================================================
Get features vectors from documents using Word2Vec and Doc2Vec
==============================================================

This is an example showing how scikit-learn can be used getting feature
vectors from a collection of documents. The two models used are Word2Vec and
Doc2Vec.
The learned vectors of the documents are saved in matrix format
[n_docs, vector_size], and are used as new features representing the
original documents.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

"""
import sys
sys.path.insert(0, '..')
import logging
from argparse import ArgumentParser

from sklearn.feature_extraction.text import Word2VecVectorizer, \
    Doc2VecVectorizer
from sklearn.datasets import fetch_20newsgroups

logging.basicConfig(level=logging.INFO)

# fetch the data
newsgroups_train = fetch_20newsgroups(subset='train')

# parse commandline arguments
parser = ArgumentParser()
parser.add_argument('-size', '--vector_size', type=int, default=100)
parser.add_argument('-min_count', '--min_count', type=int, default=5)
parser.add_argument('-window', '--window', type=int, default=5)
parser.add_argument('-iter', '--iterations', type=int, default=6)
parser.add_argument('-train', '--train', type=str)
parser.add_argument('-dw', '--dbow_words', type=bool, default=True)
parser.add_argument('-rc', '--retrain_count', type=int, default=10)
parser.add_argument('-word2vec', '--use_w2v', action='store_true')
parser.add_argument('-doc2vec', '--use_d2v', action='store_true')

usage = '''Usage: word_doc_2_vec.py -size <vector size>
           -min_count <min count of word in document> -window
           <max distance between current and predicted word>
           -iter <number of iterations for building vocabulary>
           -train <train algorithm: skip-gram/cbow for word2vec,
           pv-dm/pv-dbow/both for doc2vec> -dbow_words <True/False>
           -rc <shuffle docs and retrain model count>
           (-word2vec|-doc2vec)'''
try:
    args = parser.parse_args()
except SystemExit:
    print usage
    sys.exit(2)

# if Word2Vec model is used
if args.use_w2v:
    model = Word2VecVectorizer(vector_size=args.vector_size,
                               min_count=args.min_count,
                               window=args.window,
                               iterations=args.iterations,
                               train_algorithm=args.train)

# if Doc2Vec model is used
if args.use_d2v:
    model = Doc2VecVectorizer(vector_size=args.vector_size,
                              min_count=args.min_count,
                              window=args.window,
                              iterations=args.iterations,
                              train_algorithm=args.train,
                              dbow_words=args.dbow_words,
                              retrain_count=args.retrain_count)

vectors = model.fit_transform(newsgroups_train.data)
#print vectors
