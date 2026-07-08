"""Text representations (vectorizers) and, later, hybrid statistical features.

Phase 2 provides the three text representations compared in the study. Each is a
factory returning a *fresh, unfitted* scikit-learn vectorizer so it can be dropped
into a Pipeline and fitted on the training fold only (no leakage).

- ``bow``            : Bag-of-Words counts (CountVectorizer).
- ``tfidf_unigram``  : TF-IDF over single words.
- ``tfidf_ngram``    : TF-IDF over unigrams + n-grams (``max_n=2`` -> uni+bi,
                       ``max_n=3`` -> uni+bi+tri).

(Statistical / hybrid features are added in Phase 4.)
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bow(**kwargs) -> CountVectorizer:
    """Bag-of-Words: raw word counts."""
    return CountVectorizer(**kwargs)


def tfidf_unigram(**kwargs) -> TfidfVectorizer:
    """TF-IDF weighting over single words (unigrams)."""
    return TfidfVectorizer(ngram_range=(1, 1), **kwargs)


def tfidf_ngram(max_n: int = 2, **kwargs) -> TfidfVectorizer:
    """TF-IDF over unigrams plus n-grams up to ``max_n``.

    ``max_n=2`` gives unigrams+bigrams; ``max_n=3`` adds trigrams.
    """
    return TfidfVectorizer(ngram_range=(1, max_n), **kwargs)
