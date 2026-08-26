"""Text representations (vectorizers), statistical features, and the hybrid combiner.

Phase 2 provides the three text representations compared in the study. Each is a
factory returning a *fresh, unfitted* scikit-learn vectorizer so it can be dropped
into a Pipeline and fitted on the training fold only (no leakage).

- ``bow``            : Bag-of-Words counts (CountVectorizer).
- ``tfidf_unigram``  : TF-IDF over single words.
- ``tfidf_ngram``    : TF-IDF over unigrams + n-grams (``max_n=2`` -> uni+bi,
                       ``max_n=3`` -> uni+bi+tri).

Phase 4 adds the hybrid feature set (proposal objective 4):

- ``StatisticalFeatures`` : five cheap surface statistics per message.
- ``lexical_pipeline``    : clean -> vectorize, as one Pipeline step.
- ``hybrid``              : lexical branch + scaled statistical branch, joined.

Both Phase-4 builders take **raw** text as input. The statistical features are
computed before cleaning (cleaning is what removes the capitals and punctuation
they measure), and the lexical branch cleans internally, so the two branches see
the input they each need from a single raw-text pipeline.
"""

import re
import string

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

from .preprocessing import TextCleaner

# Matches an explicit scheme or a bare "www." host - enough for SMS text, which
# rarely contains well-formed URLs.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PUNCT_SET = set(string.punctuation)


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


class StatisticalFeatures(BaseEstimator, TransformerMixin):
    """The five surface statistics from proposal objective 4, per raw message.

    - ``char_len``      : message length in characters.
    - ``digit_density`` : share of characters that are digits (spam quotes prices,
                          short codes and phone numbers).
    - ``punct_freq``    : share of characters that are punctuation.
    - ``upper_ratio``   : share of *letters* that are uppercase (shouting). Divided
                          by letter count, not total length, so a digit-heavy
                          message is not scored as lowercase by accident.
    - ``url_count``     : number of URL-looking tokens.

    Nothing is fitted, but it is still a transformer so the downstream scaler can
    be fitted per CV fold inside a Pipeline rather than over the whole dataset.
    """

    FEATURE_NAMES = (
        "char_len",
        "digit_density",
        "punct_freq",
        "upper_ratio",
        "url_count",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = []
        for text in X:
            text = str(text)
            n_chars = len(text) or 1
            n_letters = sum(c.isalpha() for c in text) or 1
            rows.append(
                [
                    float(len(text)),
                    sum(c.isdigit() for c in text) / n_chars,
                    sum(c in _PUNCT_SET for c in text) / n_chars,
                    sum(c.isupper() for c in text) / n_letters,
                    float(len(_URL_RE.findall(text))),
                ]
            )
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.FEATURE_NAMES, dtype=object)


def lexical_pipeline(vectorizer, **clean_kwargs) -> Pipeline:
    """Clean -> vectorize, as a single step that consumes **raw** text.

    Equivalent to what notebooks 02/03 did by cleaning up front, which is what the
    Phase-4 control arm checks by reproducing the Phase-2 metrics exactly.
    """
    return Pipeline(
        [("clean", TextCleaner(**clean_kwargs)), ("vec", vectorizer)]
    )


def hybrid(vectorizer, **clean_kwargs) -> FeatureUnion:
    """Lexical features joined with the scaled statistical features.

    Consumes raw text and returns ``[vectorizer vocabulary | 5 statistics]``.

    ``MinMaxScaler`` rather than ``StandardScaler``: MultinomialNB requires
    non-negative inputs, and standardising would centre the statistics on zero and
    make roughly half of them negative. ``clip=True`` matters for the same reason -
    a test message shorter than anything in the training fold would otherwise scale
    to a negative value and break MultinomialNB at predict time.

    The scaler sits *inside* the union, so it is fitted on the training fold only,
    exactly like the vectorizer. Computing the statistics over the whole dataset
    and scaling before the split would be leakage.
    """
    return FeatureUnion(
        [
            ("lexical", lexical_pipeline(vectorizer, **clean_kwargs)),
            (
                "stats",
                Pipeline(
                    [
                        ("stats", StatisticalFeatures()),
                        ("scale", MinMaxScaler(clip=True)),
                    ]
                ),
            ),
        ]
    )
