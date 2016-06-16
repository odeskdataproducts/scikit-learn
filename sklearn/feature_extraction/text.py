# -*- coding: utf-8 -*-
# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          Robert Layton <robertlayton@gmail.com>
#          Jochen Wersdörfer <jochen@wersdoerfer.de>
#          Roman Sinayev <roman.sinayev@gmail.com>
#
# License: BSD 3 clause
"""
The :mod:`sklearn.feature_extraction.text` submodule gathers utilities to
build feature vectors from text documents.

Edited by Nan Li (nanli@odesk.com, nanli@alumni.cs.ucsb.edu)

An LdaVectorizer class and an LsiVectorizer class are added
to extract topics from text. The topic model implementation is
based on the LdaModel and LsiModel classes in the gensim.models
module.
"""
from __future__ import unicode_literals

import array
from collections import Mapping, defaultdict
import numbers
from operator import itemgetter
import re
import unicodedata

import warnings
import time
import codecs
import logging

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin

from ..externals import six
from ..externals.six.moves import xrange
from ..preprocessing import normalize
from .hashing import FeatureHasher
from .stop_words import ENGLISH_STOP_WORDS
from ..utils import deprecated
from ..utils.fixes import frombuffer_empty
from ..utils.validation import check_is_fitted

from .readability import Readability

from gensim import corpora, models
from sklearn.preprocessing import scale
from gensim.models.doc2vec import TaggedDocument


__all__ = ['CountVectorizer',
           'ENGLISH_STOP_WORDS',
           'TfidfTransformer',
           'TfidfVectorizer',
           'strip_accents_ascii',
           'strip_accents_unicode',
           'strip_tags',
           'LdaVectorizer',
           'LsiVectorizer',
           'Word2VecVectorizer',
           'Doc2VecVectorizer']

logging.basicConfig(level=logging.DEBUG,
                    format='''%(asctime)s - %(name)s
                            - %(levelname)s - %(message)s''')
logger = logging.getLogger('sklearn.feature_extraction.text')


def strip_accents_unicode(s):
    """Transform accentuated unicode symbols into their simple counterpart

    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.

    See also
    --------
    strip_accents_ascii
        Remove accentuated char for any unicode symbol that has a direct
        ASCII equivalent.
    """
    return ''.join([c for c in unicodedata.normalize('NFKD', s)
                    if not unicodedata.combining(c)])


def strip_accents_ascii(s):
    """Transform accentuated unicode symbols into ascii or nothing

    Warning: this solution is only suited for languages that have a direct
    transliteration to ASCII symbols.

    See also
    --------
    strip_accents_unicode
        Remove accentuated char for any unicode symbol.
    """
    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')


def strip_tags(s):
    """Basic regexp based HTML / XML tag stripper function

    For serious HTML/XML preprocessing you should rather use an external
    library such as lxml or BeautifulSoup.
    """
    return re.compile(r"<([^>]+)>", flags=re.UNICODE).sub(" ", s)


def _check_stop_list(stop):
    if stop == "english":
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, six.string_types):
        raise ValueError("not a built-in stop list: %s" % stop)
    else:               # assume it's a collection
        return stop


class VectorizerMixin(object):
    """Provides common code for text vectorizers (tokenization logic)."""

    _white_spaces = re.compile(r"\s\s+")

    def decode(self, doc):
        """Decode the input into a string of unicode symbols

        The decoding strategy depends on the vectorizer parameters.
        """
        if self.input == 'filename':
            with open(doc, 'rb') as fh:
                doc = fh.read()

        elif self.input == 'file':
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        if doc is np.nan:
            raise ValueError("np.nan is an invalid document, expected byte or "
                             "unicode string.")

        return doc

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens.append(" ".join(original_tokens[i: i + n]))

        return tokens

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in xrange(min_n, min(max_n + 1, text_len + 1)):
            for i in xrange(text_len - n + 1):
                ngrams.append(text_document[i: i + n])
        return ngrams

    def _char_wb_ngrams(self, text_document):
        """Whitespace sensitive char-n-gram tokenization.

        Tokenize text_document into a sequence of character n-grams
        excluding any whitespace (operating only inside word boundaries)"""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        min_n, max_n = self.ngram_range
        ngrams = []
        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in xrange(min_n, max_n + 1):
                offset = 0
                ngrams.append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams.append(w[offset:offset + n])
                if offset == 0:   # count a short word (w_len < n) only once
                    break
        return ngrams

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""
        if self.preprocessor is not None:
            return self.preprocessor

        # unfortunately python functools package does not have an efficient
        # `compose` function that would have allowed us to chain a dynamic
        # number of functions. However the cost of a lambda call is a few
        # hundreds of nanoseconds which is negligible when compared to the
        # cost of tokenizing a string of 1000 chars for instance.
        noop = lambda x: x

        # accent stripping
        if not self.strip_accents:
            strip_accents = noop
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' %
                             self.strip_accents)

        if self.lowercase:
            return lambda x: strip_accents(x.lower())
        else:
            return strip_accents

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        return lambda doc: token_pattern.findall(doc)

    def get_stop_words(self):
        """Build or fetch the effective stop words list"""
        return _check_stop_list(self.stop_words)

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()

            return lambda doc: self._word_ngrams(
                tokenize(preprocess(self.decode(doc))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(six.itervalues(vocabulary))
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in xrange(len(vocabulary)):
                    if i not in indices:
                        msg = ("Vocabulary of size %d doesn't contain index "
                               "%d." % (len(vocabulary), i))
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False

    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fit-ed)"""
        msg="%(name)s - Vocabulary wasn't fitted."
        check_is_fitted(self, 'vocabulary_', msg=msg),
        
        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")
 
    @property
    @deprecated("The `fixed_vocabulary` attribute is deprecated and will be "
                "removed in 0.18.  Please use `fixed_vocabulary_` instead.")
    def fixed_vocabulary(self):
        return self.fixed_vocabulary_


class HashingVectorizer(BaseEstimator, VectorizerMixin):
    """Convert a collection of text documents to a matrix of token occurrences

    It turns a collection of text documents into a scipy.sparse matrix holding
    token occurrence counts (or binary occurrence information), possibly
    normalized as token frequencies if norm='l1' or projected on the euclidean
    unit sphere if norm='l2'.

    This text vectorizer implementation uses the hashing trick to find the
    token string name to feature integer index mapping.

    This strategy has several advantages:

    - it is very low memory scalable to large datasets as there is no need to
      store a vocabulary dictionary in memory

    - it is fast to pickle and un-pickle as it holds no state besides the
      constructor parameters

    - it can be used in a streaming (partial fit) or parallel pipeline as there
      is no state computed during fit.

    There are also a couple of cons (vs using a CountVectorizer with an
    in-memory vocabulary):

    - there is no way to compute the inverse transform (from feature indices to
      string feature names) which can be a problem when trying to introspect
      which features are most important to a model.

    - there can be collisions: distinct tokens can be mapped to the same
      feature index. However in practice this is rarely an issue if n_features
      is large enough (e.g. 2 ** 18 for text classification problems).

    - no IDF weighting as this would render the transformer stateful.

    The hash function employed is the signed 32-bit version of Murmurhash3.

    Parameters
    ----------

    input: string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents: {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    analyzer: string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor: callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer: callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.

    ngram_range: tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words: string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

    lowercase: boolean, default True
        Convert all characters to lowercase before tokenizing.

    token_pattern: string
        Regular expression denoting what constitutes a "token", only used
        if `analyzer == 'word'`. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    n_features : integer, optional, (2 ** 20) by default
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    binary: boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype: type, optional
        Type of the matrix returned by fit_transform() or transform().

    non_negative : boolean, optional
        Whether output matrices should contain non-negative values only;
        effectively calls abs on the matrix prior to returning it.
        When True, output values can be interpreted as frequencies.
        When False, output values will have expected value zero.

    See also
    --------
    CountVectorizer, TfidfVectorizer

    """
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                 binary=False, norm='l2', non_negative=False,
                 dtype=np.float64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.binary = binary
        self.norm = norm
        self.non_negative = non_negative
        self.dtype = dtype

    def partial_fit(self, X, y=None):
        """Does nothing: this transformer is stateless.

        This method is just there to mark the fact that this transformer
        can work in a streaming setup.

        """
        return self

    def fit(self, X, y=None):
        """Does nothing: this transformer is stateless."""
        # triggers a parameter validation
        self._get_hasher().fit(X, y=y)
        return self

    def transform(self, X, y=None):
        """Transform a sequence of documents to a document-term matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, file name or file object depending on the
            constructor argument) which will be tokenized and hashed.

        y : (ignored)

        Returns
        -------
        X : scipy.sparse matrix, shape = (n_samples, self.n_features)
            Document-term matrix.

        """
        analyzer = self.build_analyzer()
        X = self._get_hasher().transform(analyzer(doc) for doc in X)
        if self.binary:
            X.data.fill(1)
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    # Alias transform to fit_transform for convenience
    fit_transform = transform

    def _get_hasher(self):
        return FeatureHasher(n_features=self.n_features,
                             input_type='string', dtype=self.dtype,
                             non_negative=self.non_negative)


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


class CountVectorizer(BaseEstimator, VectorizerMixin):
    """Convert a collection of text documents to a matrix of token counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.coo_matrix.

    If you do not provide an a-priori dictionary and you do not use an analyzer
    that does some kind of feature selection then the number of features will
    be equal to the vocabulary size found by analyzing the data.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `tokenize == 'word'`. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, optional, 1 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    binary : boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.

    See also
    --------
    HashingVectorizer, TfidfVectorizer
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df of min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype

    def _sort_features(self, X, vocabulary):
        """Sort features by name

        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(six.iteritems(vocabulary))
        map_index = np.empty(len(sorted_features), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_features):
            map_index[new_val] = old_val
            vocabulary[term] = new_val
        return X[:, map_index]

    def _limit_features(self, X, vocabulary, high=None, low=None,
                        limit=None):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.

        This does not prune samples with zero features.
        """
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = _document_frequency(X)
        tfs = np.asarray(X.sum(axis=0)).ravel()
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(six.iteritems(vocabulary)):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], removed_terms

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = _make_int_array()
        indptr = _make_int_array()
        indptr.append(0)
        for doc in raw_documents:
            for feature in analyze(doc):
                try:
                    j_indices.append(vocabulary[feature])
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        j_indices = frombuffer_empty(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.ones(len(j_indices))

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)

        X.sum_duplicates()
        return vocabulary, X

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        self._validate_vocabulary()
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(raw_documents,
                                          self.fixed_vocabulary_)

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary_:
            X = self._sort_features(X, vocabulary)

            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * n_doc)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)

            self.vocabulary_ = vocabulary

        return X

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()
 
        self._check_vocabulary()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_vocab(raw_documents, fixed_vocab=True)
        if self.binary:
            X.data.fill(1)
        return X

    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.

        Parameters
        ----------
        X : {array, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        X_inv : list of arrays, len = n_samples
            List of arrays of terms.
        """
        self._check_vocabulary()

        if sp.issparse(X):
            # We need CSR format for fast row manipulations.
            X = X.tocsr()
        else:
            # We need to convert X to a matrix, so that the indexing
            # returns 2D objects
            X = np.asmatrix(X)
        n_samples = X.shape[0]

        terms = np.array(list(self.vocabulary_.keys()))
        indices = np.array(list(self.vocabulary_.values()))
        inverse_vocabulary = terms[np.argsort(indices)]

        return [inverse_vocabulary[X[i, :].nonzero()[1]].ravel()
                for i in range(n_samples)]

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        self._check_vocabulary()

        return [t for t, i in sorted(six.iteritems(self.vocabulary_),
                                     key=itemgetter(1))]

    def load_vocabulary(self):
        """Returns vocabulary dict"""
        return [{'word': w, 'count': int(count)} for w, count in
                self.vocabulary_.iteritems()]


def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized tf or tf-idf representation

    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    The actual formula used for tf-idf is tf * (idf + 1) = tf + tf * idf,
    instead of tf * idf. The effect of this is that terms with zero idf, i.e.
    that occur in all documents of a training set, will not be entirely
    ignored. The formulas used to compute tf and idf depend on parameter
    settings that correspond to the SMART notation used in IR, as follows:

    Tf is "n" (natural) by default, "l" (logarithmic) when sublinear_tf=True.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when norm='l2', "n" (none) when norm=None.

    Parameters
    ----------
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    use_idf : boolean, optional
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean, optional
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean, optional
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    References
    ----------

    .. [Yates2011] `R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
                   Information Retrieval. Addison Wesley, pp. 68-74.`

    .. [MRS2008] `C.D. Manning, P. Raghavan and H. Schuetze  (2008).
                   Introduction to Information Retrieval. Cambridge University
                   Press, pp. 118-120.`
    """

    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log1p instead of log makes sure terms with zero idf don't get
            # suppressed entirely
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf,
                                        diags=0, m=n_features, n=n_features)

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        if hasattr(self, "_idf_diag"):
            return np.ravel(self._idf_diag.sum(axis=0))
        else:
            return None


class TfidfVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to CountVectorizer followed by TfidfTransformer.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    analyzer : string, {'word', 'char'} or callable
        Whether the feature should be made of word or character n-grams.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, default True
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `analyzer == 'word'`. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a document frequency
        strictly higher than the given threshold (corpus specific stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, optional, 1 by default
        When building the vocabulary ignore terms that have a document frequency
        strictly lower than the given threshold.
        This value is also called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.

    binary : boolean, False by default.
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs.)

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    use_idf : boolean, optional
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean, optional
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean, optional
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    idf_ : array, shape = [n_features], or None
        The learned idf vector (global term weights)
        when ``use_idf`` is set to True, None otherwise.

    See also
    --------
    CountVectorizer
        Tokenize the documents and count the occurrences of token and return
        them as a sparse matrix

    TfidfTransformer
        Apply Term Frequency Inverse Document Frequency normalization to a
        sparse matrix of occurrence counts.

    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(TfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : TfidfVectorizer
        """
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        X = super(TfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = super(TfidfVectorizer, self).transform(raw_documents)
        return self._tfidf.transform(X, copy=False)


"""
===============
Added by Nan Li
===============
"""


def _generate_gensim_corpus(count_matrix):
    """Convert a sparse-CSC word count matrix into a gensim corpus,
    which is a list of lists of tuples

    Parameters
    ----------
    count_matrix: sparse-CSC matrix
        A sparse word count matrix

    Returns
    ------
    vectors: list of lists of tuples, [[(word_id, word_count)...]...]

    See Also
    --------
    class gensim.matutils.Sparse2Corpus

    """

    # turn the sparse word count matrix into a gensim corpus (list of lists)
    count_matrix_csr = count_matrix.tocsr()
    indices = count_matrix_csr.indices
    indptr = count_matrix_csr.indptr
    nonzero_entries = count_matrix_csr.data

    corpus = []
    data_index = 0  # the index of the non-zero entries in count_matrix_csr
    for ptr in range(1, len(indptr)):
        # get the number of non-empty elements in this row (doc)
        non_empty_num_inrow = indptr[ptr] - indptr[ptr-1]
        # list of (word_id, word_#) tuples for this row (doc)
        corpus_row = []

        for i in range(non_empty_num_inrow):
            col_id = indices[data_index + i]
            corpus_row.append((col_id, nonzero_entries[data_index + i]))

        data_index += non_empty_num_inrow
        corpus.append(corpus_row)

    return corpus


def _convert_gensim_corpus2csr(corpus, num_topics=None,
                               dtype=np.float64,
                               num_docs=None, num_nnz=None):
    """
    Convert a gensim corpus into a sparse matrix,
    in scipy.sparse.csr_matrix format, with documents as rows

    If the number of terms, documents and non-zero elements is known,
    you can pass them here as parameters and a more memory efficient
    code path will be taken.

    The code in this method is copied and modified from
    gensim.mathutils.corpus2csc method.

    Parameters
    ----------
    corpus: list of lists of tuples
        [[(word_id, word_count)...]...]

    num_topics: integer
        Number of topics

    num_docs: integer
        Number of documents

    num_nnz: integer
        Number of non-zero entries in the corpus

    Returns
    ------
    vectors: sparse matrix (scipy.sparse.csr_matrix), [n_docs, n_topics]

    """

    if (num_topics is not None
       and num_docs is not None
       and num_nnz is not None):
        # faster and much more memory-friendly version
        # of creating the sparse csc
        posnow, indptr = 0, [0]
        # HACK assume feature ids fit in 32bit integer
        indices = np.empty((num_nnz,), dtype=np.int32)
        data = np.empty((num_nnz,), dtype=dtype)
        for _, doc in enumerate(corpus):
            posnext = posnow + len(doc)
            indices[posnow: posnext] = [feature_id for feature_id, _ in doc]
            data[posnow: posnext] = [feature_weight
                                     for _, feature_weight in doc]
            indptr.append(posnext)
            posnow = posnext
        assert posnow == num_nnz, '''mismatch between supplied
                                     and computed number of non-zeros'''
        result = sp.csr_matrix((data, indices, indptr),
                               shape=(num_docs, num_topics),
                               dtype=dtype)
    else:
        # slower version
        # determine the sparse matrix parameters during iteration
        num_nnz, data, indices, indptr = 0, [], [], [0]
        for _, doc in enumerate(corpus):
            indices.extend([feature_id for feature_id, _ in doc])
            data.extend([feature_weight for _, feature_weight in doc])
            num_nnz += len(doc)
            indptr.append(num_nnz)
        if num_topics is None:
            num_topics = max(indices) + 1 if indices else 0
        num_docs = len(indptr) - 1
        # now num_docs, num_topics and num_nnz contain the correct values
        data = np.asarray(data, dtype=dtype)
        indices = np.asarray(indices)
        result = sp.csr_matrix((data, indices, indptr),
                               shape=(num_docs, num_topics),
                               dtype=dtype)
    return result


class LdaVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of LDA topic features

    Latent dirichlet allocation (LDA) is a widely-used generative model to
    extract latent topics from a collection of documents.
    Each document is modeled as a distribution over a set of topics,
    and each topic is modeled as a distribution over a set of keywords.

    The LdaModel from gensim is used as the LDA implementation.

    Parameters
    ----------
    == Parameters related to CountVectorizer ==
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `tokenize == 'word'`. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, optional, 1 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    binary : boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().


    == Parameters related to LdaVectorizer ==
    num_topics : integer
        Number of requested latent topics.

    id2word : gensim.corpora.Dictionary
        A mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as
        for debugging and topic printing.

    alpha : float or vector, optional
        Hyperpameter for the symmetric Dirichlet prior on the
        document-topic distribution, or can be set to a vector
        for asymmetric prior.

    eta : float or vector, optional
        Hyperparameter for the symmetric Dirichlet prior on the
        topic-word distribution, or can be set to a vector
        for asymmetric prior.

    distributed :  boolean, optional
        Turned on to force distributed computing (see the web tutorial
        on how to set up a cluster of machines for gensim).

    topic_file : string, optional, 'lda_topics.txt' by default
        The log file used to record all learned topics


    See also
    --------
    CountVectorizer
        Tokenize the documents and count the occurrences of token and return
        them as a sparse matrix

    gensim.models.LdaModel
        Encapsulate functionality for the
        Latent Dirichlet Allocation algorithm

    References
    ----------

    .. [HoffmanBB10] `Matthew D. Hoffman, David M. Blei, Francis R. Bach:
    Online Learning for Latent Dirichlet Allocation. NIPS 2010: 856-864`

    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words='english', token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None, 
                 vocabulary=None, binary=False, dtype=np.int64,
                 num_topics=100,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha=None, eta=None,
                 decay=0.5,
                 writedown_topics=True,
                 topic_file='lda_topics.txt'):

        # initialize a CountVectorizer object
        super(LdaVectorizer, self).__init__(
            input=input,
            encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=False,
            dtype=dtype)

        self.num_topics = num_topics
        self.distributed = distributed
        self.chunksize = chunksize
        self.passes = passes
        self.update_every = update_every
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.writedown_topics = writedown_topics
        self.topic_file = topic_file

    def fit(self, raw_docs, y=None):
        """Learn a conversion law from raw documents
        to array data (topic weights)"""

        # create word-count matrix (sparse csc) using
        # the parent CountVectorizer
        count_matrix = super(LdaVectorizer, self).fit_transform(raw_docs)
        logger.info('Building LDA model: sklearn word count matrix completed!')

        # build a gensim dictionary
        self._dictionary = corpora.Dictionary([super(LdaVectorizer, self)
                                              .get_feature_names()])

        logger.info('Building LDA model: gensim Dictionary generated!')

        self._corpus = _generate_gensim_corpus(count_matrix)
        logger.info('Building LDA model: gensim corpus generated!')

        # build a gensim tf-idf model
        #self._model_tfidf = models.TfidfModel(self._corpus)
        # turn the word count into a normalized tfidf
        #self._corpus_tfidf = self._model_tfidf[self._corpus]
        #logger.info('Building LDA model: gensim tf-idf corpus generated!')

        #logger.info('Cleaning up the TFIDF model...')
        #self._model_tfidf = None

        # build an LDA model
        # Specifying dictionary is important if we want to output actual words,
        # not IDs, into the topic log file
        self._model_lda = models.LdaModel(
            #corpus=self._corpus_tfidf, id2word=self._dictionary,
            corpus=self._corpus, id2word=self._dictionary,
            num_topics=self.num_topics,
            distributed=self.distributed,
            chunksize=self.chunksize, passes=self.passes,
            update_every=self.update_every,
            alpha=self.alpha, eta=self.eta,
            decay=self.decay)
        logger.info('Building LDA model: gensim LDA model generated!')

        #self._corpus_lda = self._model_lda[self._corpus]

        logger.info('Cleaning up gensim corpus...')
        self._corpus = None
        #self._corpus_tfidf = None
        self._dictionary = None

        if self.writedown_topics:
            lda_topic_file = codecs.open(self.topic_file, 'a', 'utf-8')
            lda_topic_file.write('\n\n======= '+str(time.ctime())+' =======\n')
            topic_counter = 0
            for i in range(0, self.num_topics):
                lda_topic_file.write('\ntopic #'+str(topic_counter) + ': '
                                     + self._model_lda.print_topic(i, 20)
                                     + '\n')
                topic_counter += 1
                lda_topic_file.flush()
            lda_topic_file.flush()
            logger.info('Finished logging!')

        return self

    def fit_transform(self, raw_docs, y=None):
        """Learn the LDA model from the raw documents and
        return the LDA vector representation of the documents

        Parameters
        ----------
        raw_docs : iterable
            A collection of documents, each is represented as a string

        Returns
        -------
        vectors : array, [n_samples, n_topics]
        """
        self.fit(raw_docs)
        count_matrix = super(LdaVectorizer, self).fit_transform(raw_docs)
        corpus = _generate_gensim_corpus(count_matrix)
        return _convert_gensim_corpus2csr(self._model_lda[corpus],
                                          num_topics=self.num_topics)

        #return _convert_gensim_corpus2csr(self._model_lda[self._corpus])
        #return _convert_gensim_corpus2csr(self._corpus_lda)

    def transform(self, raw_docs):
        """Return the LDA vector representation of the new documents
        using the learned LDA model

        Parameters
        ----------
        raw_docs : iterable
            A collection of documents, each is represented as a string

        Returns
        -------
        vectors : array, [n_samples, n_topics]
        """

        # create word-count matrix (sparse csc) using the
        # parent CountVectorizer
        count_matrix = super(LdaVectorizer, self).transform(raw_docs)
        logger.info('''Transforming new docs:
                     sklearn word count matrix completed!''')

        corpus = _generate_gensim_corpus(count_matrix)
        logger.info('Transforming new docs: gensim corpus generated!')

        return _convert_gensim_corpus2csr(self._model_lda[corpus],
                                          num_topics=self.num_topics)

    def load_vocabulary(self):
        out_result = []
        for i in range(0, self.num_topics):
            out_result.append({
                'topic': 'topic #%s' % i,
                'content': [(str(k[0]), float(k[1])) for k in
                            self._model_lda.show_topic(i, 20)]})
        return out_result


class LsiVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of LSI topic features

    Latent semantic analysis/indexing (LSA/LSI) is a widely-used technique to
    analyze documents and find the unerlying meaning or concepts of
    those documents. LSA assumes that words that are close in meaning
    will occur in similar pieces of text. A matrix containing word counts
    per document is constructed from a corpus of documents and a linear
    algebra technique called singular value decomposition (SVD) is used to
    reduce the number of words while preserving the similarity structure
    among documents.

    The LsiModel from gensim is used as the LSI implementation.

    Parameters
    ----------
    == Parameters related to CountVectorizer ==
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `tokenize == 'word'`. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, optional, 1 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    binary : boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().


    == Parameters related to LsiVectorizer ==
    num_topics : integer
        Number of requested latent topics.

    id2word : gensim.corpora.Dictionary
        A mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging
        and topic printing.

    distributed :  boolean, optional
        Turned on to force distributed computing (see the web tutorial
        on how to set up a cluster of machines for gensim).

    onepass : boolean, optional
        Turned off to force a multi-pass stochastic algorithm.

    power_iters : integer, optional
        Increasing the number of power iterations improves accuracy,
        but lowers performance.

    extra_samples : integer, optional

    `power_iters` and `extra_samples` affect the accuracy of the stochastic
    multi-pass algorithm, which is used either internally (`onepass=True`) or
    as the front-end algorithm (`onepass=False`).

    topic_file : string, optional, 'lsi_topics.txt' by default
        The log file used to record all learned topics

    See also
    --------
    CountVectorizer
        Tokenize the documents and count the occurrences of token and return
        them as a sparse matrix

    gensim.models.LsiModel
        Encapsulate functionality for the Latent Semantic Indexing algorithm

    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words='english', token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64,
                 num_topics=100,
                 distributed=False, chunksize=20000,
                 onepass=True, power_iters=2, extra_samples=100,
                 decay=1.0,
                 writedown_topics=True,
                 topic_file='lsi_topics.txt'):

        # initialize a CountVectorizer object
        super(LsiVectorizer, self).__init__(
            input=input,
            encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary,
            binary=False, dtype=dtype)

        self.num_topics = num_topics
        self.distributed = distributed
        self.chunksize = chunksize
        self.onepass = onepass
        self.power_iters = power_iters
        self.extra_samples = extra_samples
        self.decay = decay
        self.writedown_topics = writedown_topics
        self.topic_file = topic_file

    def fit(self, raw_docs, y=None):
        """Learn a conversion law from raw documents
        to array data (topic weights)"""

        # create word-count matrix (sparse csc) using
        # the parent CountVectorizer
        count_matrix = super(LsiVectorizer, self).fit_transform(raw_docs)
        logger.info('Building LSI model: sklearn word count matrix completed!')

        # build a gensim dictionary
        self._dictionary = corpora.Dictionary([super(LsiVectorizer, self)
                                              .get_feature_names()])
        logger.info('Building LSI model: gensim Dictionary generated!')

        self._corpus = _generate_gensim_corpus(count_matrix)
        logger.info('Building LSI model: gensim corpus generated!')

        # build a gensim tf-idf model
        self._model_tfidf = models.TfidfModel(self._corpus)
        # turn the word count into a normalized tfidf
        self._corpus_tfidf = self._model_tfidf[self._corpus]
        logger.info('Building LSI model: gensim tf-idf corpus generated!')

        #logger.info('Cleaning up the TFIDF model...')
        #self._model_tfidf = None

        # build an LSI model
        # Specifying dictionary is important if we want to output actual words,
        # not IDs, into the topic log file
        self._model_lsi = models.LsiModel(
            corpus=self._corpus_tfidf, id2word=self._dictionary,
            num_topics=self.num_topics,
            distributed=self.distributed, chunksize=self.chunksize,
            onepass=self.onepass, power_iters=self.power_iters,
            extra_samples=self.extra_samples,
            decay=self.decay)
        logger.info('Building LSI model: gensim LSI model generated!')

        #self._corpus_lsi = self._model_lsi[self._corpus]

        logger.info('Cleaning up gensim corpus...')
        self._corpus = None
        self._corpus_tfidf = None
        self._dictionary = None

        if self.writedown_topics:
            lsi_topic_file = codecs.open(self.topic_file, 'a', 'utf-8')
            lsi_topic_file.write('\n\n======= '+str(time.ctime())+' =======\n')
            topic_counter = 0
            # -1 means show all topics
            for topic in self._model_lsi.show_topics(-1, 20):
                lsi_topic_file.write('\ntopic #' + str(topic_counter)
                                     + ': ' + topic[1] + '\n')
                topic_counter += 1
                lsi_topic_file.flush()
            lsi_topic_file.flush()
            logger.info('Finished logging!')

        return self

    def fit_transform(self, raw_docs, y=None):
        """Learn the LSI model from the raw documents and
        return the LSI vector representation of the documents

        Parameters
        ----------
        raw_docs : iterable
            A collection of documents, each is represented as a string

        Returns
        -------
        vectors : array, [n_samples, n_topics]
        """

        self.fit(raw_docs)
        count_matrix = super(LsiVectorizer, self).fit_transform(raw_docs)
        corpus = _generate_gensim_corpus(count_matrix)
        corpus_tfidf = self._model_tfidf[corpus]

        #return _convert_gensim_corpus2csr(self._model_lsi[corpus],
        #                                  num_topics=self.num_topics)
        return _convert_gensim_corpus2csr(self._model_lsi[corpus_tfidf],
                                          num_topics=self.num_topics)

        #return _convert_gensim_corpus2csr(self._model_lsi[self._corpus])
        #return _convert_gensim_corpus2csr(self._corpus_lsi)

    def transform(self, raw_docs):
        """Return the LSI vector representation of the new documents
        using the learned LSI model

        Parameters
        ----------
        raw_docs : iterable
            A collection of documents, each is represented as a string

        Returns
        -------
        vectors : array, [n_samples, n_topics]
        """

        # create word-count matrix (sparse csc) using
        # the parent CountVectorizer
        count_matrix = super(LsiVectorizer, self).transform(raw_docs)
        logger.info('''Transforming new docs:
                    sklearn word count matrix completed!''')

        corpus = _generate_gensim_corpus(count_matrix)
        corpus_tfidf = self._model_tfidf[corpus]
        logger.info('Transforming new docs: gensim corpus generated!')

        #return _convert_gensim_corpus2csr(self._model_lsi[corpus],
        #                                  num_topics=self.num_topics)
        return _convert_gensim_corpus2csr(self._model_lsi[corpus_tfidf],
                                          num_topics=self.num_topics)

    def load_vocabulary(self):
        out_result = []
        i = 0
        # -1 means show all topics
        for topic in self._model_lsi.show_topics(-1, 20, formatted=False):
            out_result.append({
                'topic': 'topic #%s' % i,
                'content': [(str(k[0]), float(k[1])) for k in topic[1]]
            })
            i += 1
        return out_result


class ReadabilityTransformer():
    """Convert a collection of raw documents to a vector of readability scores

    Parameters
    ----------
    readability_type : string {'ari', 'flesch_reading_ease',
                    'flesch_kincaid_grade_level', 'gunning_fog_index',
                    'smog_index', 'coleman_liau_index', 'lix', 'rix'},
                    optional, 'smog_index' by default
        Specify the type of the readability score used for transformation

    """

    def __init__(self, readability_type='smog_index'):
        self.readability_types = {'ari': self.ari,
                'flesch_reading_ease': self.flesch_reading_ease,
                'flesch_kincaid_grade_level': self.flesch_kincaid_grade_level,
                'gunning_fog_index': self.gunning_fog_index,
                'smog_index': self.smog_index,
                'coleman_liau_index': self.coleman_liau_index,
                'lix': self.lix,
                'rix': self.rix}
        if readability_type not in self.readability_types:
            readability_type = 'smog_index'
        self.readability_type = readability_type

    def transform(self, raw_docs):
        """Return the readability scores of the given set of documents

        Parameters
        ----------
        raw_docs : iterable
            A collection of documents, each is represented as a string

        Returns
        -------
        vector : array, [n_samples, 1]
        """
        scores = []
        indices = []
        indptr = []
        count = 0
        for doc in raw_docs:
            r = Readability(doc)
            score = self.readability_types[self.readability_type](r)
            scores.append(score)
            indices.append(0)
            indptr.append(count)
            count += 1

        indptr.append(count)
        num_docs = len(raw_docs)
        return sp.csr_matrix((np.array(scores),
                              np.array(indices),
                              np.array(indptr)),
                              shape=(num_docs, 1))

    def ari(self, r):
        return r.ARI()

    def flesch_reading_ease(self, r):
        return r.FleschReadingEase()

    def flesch_kincaid_grade_level(self, r):
        return r.FleschKincaidGradeLevel()

    def gunning_fog_index(self, r):
        return r.GunningFogIndex()

    def smog_index(self, r):
        return r.SMOGIndex()

    def coleman_liau_index(self, r):
        return r.ColemanLiauIndex()

    def lix(self, r):
        return r.LIX()

    def rix(self, r):
        return r.RIX()


class Word2VecVectorizer(BaseEstimator, VectorizerMixin):
    """Converts a collection of text documents to a matrix:
    gaussian distribution with a mean of zero, meaning that values above
    the mean will be positive, and those below the mean will be negative.

    The word2vec provides an efficient implementation of the continuous
    bag-of-words and skip-gram architectures for computing vector
    representations of words. It takes a text corpus as input and produces
    the word vectors as output. It first constructs a vocabulary from the
    training text data and then learns vector representation of words.
    The resulting word vector can be used as features in many natural language
    processing and machine learning applications.

    The Word2Vec model from gensim is used for implementation.

    More info:
    http://rare-technologies.com/word2vec-tutorial/
    https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `tokenize == 'word'`. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, optional, 1 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    binary : boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    train_algorithm : string, optional, 'skip-gram' by default.
        Defines the training algorithm. Otherwise, 'cbow' is employed.

    vector_size : int, optional. 100 by default
        Dimensionality of the feature vectors.

    window : int, optional. 5 by default
        Maximum distance between the current and predicted word within
        a sentence.

    alpha : float, optional. 0.025 by default
        Initial learning rate (will linearly drop to zero as training
        progresses).

    seed : string or int or float or boolean, optional. 1 by default.
        Used for the random number generator. Initial vectors for each
        word are seeded with a hash of the concatenation of word + str(seed).

    min_count : int, optional. 5 by default.
        Ignore all words with total frequency lower than this.

    max_vocab_size : int, optional. None by default.
        Limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million
        word types need about 1GB of RAM. Set to `None` for no limit (default).

    sample : float, optional. 0 by default
        Threshold for configuring which higher-frequency words are randomly
        downsampled; default is 0 (off), useful value is 1e-5.

    workers : int, optional. 1 by default.
        Use this many worker threads to train the model (=faster training
        with multicore machines).

    hierarchical_sampling : boolean, True by default
        Hierarchical sampling will be used for model training if true.

    negative_sampling : int, optional. 0 by default.
        If > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20)

    cbow_mean : boolean, False by default.
        If false (default), use the sum of the context word vectors.
        If true, use the mean. Only applies when cbow is used.

    hash_function : string, optional. 'hash' by default
        Hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in
        hash function.

    iterations : int, optional. 1 by default
        Number of iterations (epochs) over the corpus.

    trim_rule : string, optional. None by default
        Vocabulary trimming rule, specifies whether certain words should remain
        in the vocabulary, be trimmed away, or handled using the default
        (discard if word count < min_count). Can be None (min_count will be
        used), or a callable that accepts parameters (word, count, min_count)
        and returns either util.RULE_DISCARD, util.RULE_KEEP or
        util.RULE_DEFAULT.
        Note: The rule, if given, is only used prune vocabulary during
        build_vocab() and is not stored as part of the model.

    sorted_vocab : int, optional.
        If 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', binary=False,
                 dtype=np.int64,

                 # Word2Vec model parameters
                 train_algorithm='skip-gram', vector_size=100, min_count=5,
                 max_vocab_size=None, window=5, sample=0, seed=1, workers=1,
                 min_alpha=0.0001, hierarchical_sampling=True, iterations=1,
                 negative_sampling=False, cbow_mean=False, null_word=False,
                 hash_function='hash', trim_rule=None, sorted_vocab=True,
                 alpha=0.025):

        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.binary = binary
        self.dtype = dtype

        # these values are predefined, because Word2Vec model builds
        # the dictionary itself and takes corresponding values from
        # min_count and max_vocab_size parameters
        self.max_df = 1.0
        self.min_df = 1
        self.max_features = None
        self.vocabulary = None

        self.vector_size = vector_size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.train_algorithm = train_algorithm
        self.hierarchical_sampling = hierarchical_sampling
        self.negative_sampling = negative_sampling
        self.cbow_mean = cbow_mean
        self.hash_function = hash_function
        self.iterations = iterations
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab

    def fit(self, raw_documents, y=None):
        """Build and learn the model of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """
        logger.info('Prepare sentences from documents')
        analyze = self.build_analyzer()
        self.sentences = []
        for doc in raw_documents:
            self.sentences.append(analyze(doc))

        logger.info('Build and train the model')
        # build and train the model (model builds an internal vocabulary
        # from sentences, trains itself and creates vectors for each word)
        logger.info('Training algorithm - %s' % self.train_algorithm)
        self._model_word2vec = models.Word2Vec(
            self.sentences, min_count=self.min_count, size=self.vector_size,
            max_vocab_size=self.max_vocab_size, alpha=self.alpha,
            window=self.window, sample=self.sample, seed=self.seed,
            workers=self.workers, min_alpha=self.min_alpha,
            sg=1 if self.train_algorithm == 'skip-gram' else 0,
            hs=self.hierarchical_sampling, negative=self.negative_sampling,
            cbow_mean=self.cbow_mean, hashfxn=eval(self.hash_function),
            null_word=self.null_word, trim_rule=self.trim_rule,
            sorted_vocab=self.sorted_vocab, iter=self.iterations)

        return self

    def fit_transform(self, raw_documents, y=None):
        """Build the model and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, vector_size]
            Document-term matrix.
        """
        self.fit(raw_documents)
        logger.info('Get vectors ... ')
        train_vecs = np.concatenate([self._build_doc_vector(sentence)
                                     for sentence in self.sentences])
        return scale(train_vecs)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : matrix, [n_samples, vector_size]
            Document-term matrix.
        """
        logger.info('Get vectors ... ')
        analyze = self.build_analyzer()
        train_vecs = np.concatenate([self._build_doc_vector(analyze(doc))
                                     for doc in raw_documents])
        return scale(train_vecs)

    def _build_doc_vector(self, doc):
        """
        Builds doc vector by using the average value of all word vectors
        in the sentence (text)
        """
        if not self._model_word2vec:
            raise ValueError("Build and train the model first")

        if not len(self._model_word2vec.vocab):
            raise ValueError("Empty vocabulary; perhaps the documents only "
                             "contain stop words or min count of word "
                             "presence in documents is too high")

        vec = np.zeros(self.vector_size).reshape((1, self.vector_size))
        count = 0.
        for word in doc:
            try:
                vec += self._model_word2vec[word].reshape((1,
                                                           self.vector_size))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    def load_vocabulary(self):
        """
        Returns vocabulary dict
        """
        from six import iteritems
        vocabulary = self._model_word2vec.vocab
        return_dict = []
        for word, vocab in sorted(iteritems(vocabulary),
                                  key=lambda item: -item[1].count):
            return_dict.append({
                'word': str(word),
                'count': int(vocab.count),
                'vector': [float(v) for v in
                           self._model_word2vec.syn0[vocab.index]]})
        return return_dict


class Doc2VecVectorizer(BaseEstimator, VectorizerMixin):
    """Converts a collection of text documents to a matrix:
    gaussian distribution with a mean of zero, meaning that values above
    the mean will be positive, and those below the mean will be negative.

    Doc2vec (aka paragraph2vec, aka sentence embeddings) modifies the
    word2vec algorithm to unsupervised learning of continuous representations
    for larger blocks of text, such as sentences, paragraphs or entire
    documents. In the word2vec architecture, the two algorithm names are
    “continuous bag of words” (cbow) and “skip-gram” (sg); in the doc2vec
    architecture, the corresponding algorithms are “distributed memory” (dm)
    and “distributed bag of words” (dbow).

    The Doc2Vec model from gensim is used for implementation.

    More info:
    http://rare-technologies.com/doc2vec-tutorial/
    https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis

    See also: Word2VecVectorizer

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `tokenize == 'word'`. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    max_df : float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, optional, 1 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    binary : boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    train_algorithm : string, optional, 'PV-DM' (distributed memory) by default.
        Defines the training algorithm. Otherwise, 'PV-DBOW' (distributed bag
        of words) is employed.

    vector_size : int, optional. 300 by default
        Dimensionality of the feature vectors.

    window : int, optional. 8 by default
        Maximum distance between the current and predicted word within
        a sentence.

    alpha : float, optional. 0.025 by default
        Initial learning rate (will linearly drop to zero as training
        progresses).

    seed : string or int or float or boolean, optional. 1 by default.
        Used for the random number generator. Only runs with a single worker
        will be deterministically reproducible because of the ordering
        randomness in multi-threaded runs.

    min_count : int, optional. 5 by default.
        Ignore all words with total frequency lower than this.

    max_vocab_size : int, optional. None by default.
        Limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million
        word types need about 1GB of RAM. Set to `None` for no limit (default).

    sample : float, optional. 0 by default
        Threshold for configuring which higher-frequency words are randomly
        downsampled; default is 0 (off), useful value is 1e-5.

    workers : int, optional. 1 by default.
        Use this many worker threads to train the model (=faster training
        with multicore machines).

    hierarchical_sampling : boolean, True by default
        Hierarchical sampling will be used for model training if true.

    negative_sampling : int, optional. 0 by default.
        If > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20)

    hash_function : string, optional. 'hash' by default
        Hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in
        hash function.

    iterations : int, optional. 1 by default
        Number of iterations (epochs) over the corpus.

    trim_rule : string, optional. None by default
        Vocabulary trimming rule, specifies whether certain words should remain
        in the vocabulary, be trimmed away, or handled using the default
        (discard if word count < min_count). Can be None (min_count will be
        used), or a callable that accepts parameters (word, count, min_count)
        and returns either util.RULE_DISCARD, util.RULE_KEEP or
        util.RULE_DEFAULT.
        Note: The rule, if given, is only used prune vocabulary during
        build_vocab() and is not stored as part of the model.

    sorted_vocab : int, optional.
        If 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.

    dm_mean : boolean, False by default
        if false, use the sum of the context word vectors. If true, use the mean
        Only applies when dm is used in non-concatenative mode.

    dm_concat : boolean. False by default
        If True, use concatenation of context vectors rather than sum/average;
        Note concatenation results in a much-larger model, as the input
        is no longer the size of one (sampled or arithmatically combined)
        word vector, but the size of the tag(s) and all words in the context
        strung together.

    dm_tag_count : integer, 1 by default
        Expected constant number of document tags per document, when using
        dm_concat mode

    dbow_words : boolean, default is False.
        If set to True trains word-vectors (in skip-gram fashion) simultaneous
        with DBOW doc-vector training; default is False (faster training of
        doc-vectors only)
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, analyzer='word', binary=False,
                 token_pattern=r"(?u)[\b\w\w+\b]+|[.,!?();:\"{}\[\]]",
                 ngram_range=(1, 1), dtype=np.int64,

                 # Word2Vec model parameters
                 train_algorithm='pv-dm', vector_size=300, min_count=5,
                 max_vocab_size=None, window=8, sample=0, seed=1, workers=1,
                 min_alpha=0.0001, hierarchical_sampling=True, iterations=1,
                 negative_sampling=False, hash_function='hash',
                 trim_rule=None, sorted_vocab=True, alpha=0.025,
                 dbow_words=False, dm_mean=False, dm_concat=False,
                 dm_tag_count=1, docvecs=None, docvecs_mapfile=None,
                 comment=None, retrain_count=10):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.binary = binary
        self.dtype = dtype

        # these values are predefined, because Word2Vec model builds
        # the dictionary itself and takes corresponding values from
        # min_count and max_vocab_size parameters
        self.max_df = 1.0
        self.min_df = 1
        self.max_features = None
        self.vocabulary = None

        self.vector_size = vector_size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.train_algorithm = train_algorithm
        self.hierarchical_sampling = hierarchical_sampling
        self.negative_sampling = negative_sampling
        self.hash_function = hash_function
        self.iterations = iterations
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab

        self.dbow_words = dbow_words
        self.dm_mean = dm_mean
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.docvecs = docvecs
        self.docvecs_mapfile = docvecs_mapfile
        self.comment = comment
        self.retrain_count = retrain_count

        self.tagged_documents = []

    def fit(self, raw_documents, y=None):
        """Build and learn the model of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """
        logger.info('Prepare documents')
        self._prepare_documents(raw_documents)

        # build and train the model (model builds an internal vocabulary
        # from sentences, trains itself and creates vectors for each doc)
        logger.info('Training algorithm - %s' % self.train_algorithm)
        self._model_doc2vec_dm = None
        self._model_doc2vec_dbow = None

        if self.train_algorithm in ('both', 'pv-dm'):
            self._model_doc2vec_dm = models.Doc2Vec(
                documents=self.tagged_documents, min_count=self.min_count,
                size=self.vector_size, max_vocab_size=self.max_vocab_size,
                alpha=self.alpha, window=self.window, sample=self.sample,
                seed=self.seed, workers=self.workers, min_alpha=self.min_alpha,
                dm=1,  # training algorithm
                hs=self.hierarchical_sampling, negative=self.negative_sampling,
                hashfxn=eval(self.hash_function), trim_rule=self.trim_rule,
                sorted_vocab=self.sorted_vocab, iter=self.iterations,
                dbow_words=self.dbow_words, dm_mean=self.dm_mean,
                dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
                docvecs=self.docvecs, docvecs_mapfile=self.docvecs_mapfile,
                comment=self.comment)
        if self.train_algorithm in ('both', 'pv-dbow'):
            self._model_doc2vec_dbow = models.Doc2Vec(
                documents=self.tagged_documents, min_count=self.min_count,
                size=self.vector_size, max_vocab_size=self.max_vocab_size,
                alpha=self.alpha, window=self.window, sample=self.sample,
                seed=self.seed, workers=self.workers, min_alpha=self.min_alpha,
                dm=0,  # training algorithm
                hs=self.hierarchical_sampling, negative=self.negative_sampling,
                hashfxn=eval(self.hash_function), trim_rule=self.trim_rule,
                sorted_vocab=self.sorted_vocab, iter=self.iterations,
                dbow_words=self.dbow_words, dm_mean=self.dm_mean,
                dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
                docvecs=self.docvecs, docvecs_mapfile=self.docvecs_mapfile,
                comment=self.comment)
        logger.info('Model was built and trained')

        # pass through the data set multiple times, shuffling the docs
        # each time to improve accuracy
        self._shuffle_and_train()
        logger.info('Training finished')

        return self

    def fit_transform(self, raw_documents, y=None):
        """Build the model and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, vector_size]
            Document-term matrix.
        """
        self.fit(raw_documents)
        logger.info('Get vectors ...')
        return self._get_vectors(self.tagged_documents)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : matrix, [n_samples, vector_size]
            Document-term matrix.
        """
        logger.info('Prepare sentences passed to transform for training')
        self._prepare_documents(raw_documents)
        self._shuffle_and_train()
        logger.info('Training finished. Get vectors ...')

        return self._get_vectors(self.tagged_documents)

    def _get_vectors(self, tagged_docs):
        """
        Get vectors learned by the model
        """
        if not (self._model_doc2vec_dm or self._model_doc2vec_dbow):
            raise ValueError("Build and train the model first")

        if self._model_doc2vec_dm:
            vectors_dm = [np.array(
                self._model_doc2vec_dm.docvecs[doc.tags[0]]).reshape(
                    (1, self.vector_size))
                for doc in tagged_docs]
            v_dm = np.concatenate(vectors_dm)  # result for DM algorithm

        if self._model_doc2vec_dbow:
            vectors_dbow = [np.array(
                self._model_doc2vec_dbow.docvecs[doc.tags[0]]).reshape(
                    (1, self.vector_size))
                for doc in tagged_docs]
            v_dbow = np.concatenate(vectors_dbow)  # result for DBOW algorithm

        if self.train_algorithm == 'both':
            return np.hstack((v_dm, v_dbow))
        elif self.train_algorithm == 'pv-dm':
            return v_dm
        else:
            return v_dbow

    def _prepare_documents(self, raw_documents, label='tag'):
        """
        Transforms raw_documents to TaggedDocuments
        """
        analyze = self.build_analyzer()
        self.tagged_documents = []
        i = 0
        for doc in raw_documents:
            tag = '%s_%s' % (label, i)
            self.tagged_documents.append(
                TaggedDocument(analyze(doc), [tag]))
            i += 1

        return self.tagged_documents

    def _shuffle_and_train(self):
        """
        Shuffles the tagged documents list and re-trains the model
        """
        if not (self._model_doc2vec_dm or self._model_doc2vec_dbow):
            raise ValueError("Build and train the model first")

        for epoch in range(self.retrain_count):
            logger.info('Shuffle the docs and train the model again. Step %s '
                        'of %s' % (epoch + 1, self.retrain_count))
            perm = np.random.permutation(self.tagged_documents)
            td = []
            for i in perm:
                td.append(TaggedDocument(i[0], i[1]))
            del perm
            if self._model_doc2vec_dm:
                self._model_doc2vec_dm.train(td)
            if self._model_doc2vec_dbow:
                self._model_doc2vec_dbow.train(td)

    def load_vocabulary(self):
        """
        Outputs vocabulary
        """
        return_dict = []
        for doc in self.tagged_documents:
            res = {
                'tags': [str(tag) for tag in doc.tags],
                'sentence': [str(word) for word in doc.words],
            }
            vecs_dm = []
            vecs_dbow = []
            for tag in doc.tags:
                if self._model_doc2vec_dm:
                    vecs_dm.append([float(v) for v in
                                   self._model_doc2vec_dm.docvecs[tag]])
                if self._model_doc2vec_dbow:
                    vecs_dbow.append([float(v) for v in
                                     self._model_doc2vec_dbow.docvecs[tag]])
            if len(vecs_dm):
                res['vectors_pv_dm'] = vecs_dm
            if len(vecs_dbow):
                res['vectors_pv_dbow'] = vecs_dbow
            return_dict.append(res)
        return return_dict
