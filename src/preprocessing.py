"""Data loading, text cleaning, and the single persisted train/test split.

Public API:
- ``load_dataset``  : read the CSV, tidy columns, map labels, drop duplicates.
- ``clean``         : normalise a message; each step is independently toggleable
                      (Phase 3's ablation flips these switches).
- ``TextCleaner``   : ``clean`` wrapped as a Pipeline step (Phase 4 needs the raw
                      text to reach the statistical-feature branch, so cleaning
                      moves inside the pipeline rather than being applied first).
- ``make_split``    : create the ONE stratified 80/20 split, persisted to disk
                      so every notebook reuses the exact same split.
"""

import re
import string

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, HAM, SPAM, RANDOM_SEED, SPLIT_PATH, TEST_SIZE

# --- Lazy NLTK resources (stopwords list + Porter stemmer) -----------------
# Imported lazily so importing this module never fails if NLTK data is missing;
# the helpers download it on first use.
_STEMMER = None
_STOPWORDS = None


def _get_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        from nltk.corpus import stopwords

        try:
            _STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            import nltk

            nltk.download("stopwords")
            _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def _get_stemmer():
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer

        _STEMMER = PorterStemmer()
    return _STEMMER


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_TOKEN_RE = re.compile(r"\b\w+\b")
# Keeps punctuation as standalone tokens: a run of word chars, OR a single
# non-word/non-space char. Used only when ``remove_punct=False`` so the "keep
# punctuation" ablation config genuinely retains "!", "£", "$" etc. as features.
_TOKEN_RE_KEEP_PUNCT = re.compile(r"\w+|[^\w\s]")


def load_dataset(drop_duplicates: bool = True) -> pd.DataFrame:
    """Load the SMS Spam Collection as a tidy DataFrame with columns ``label``,``text``.

    - Maps ``ham -> 0`` and ``spam -> 1`` (positive class = spam).
    - **Reconstructs** the ~50 messages whose text was split across the extra
      ``Unnamed: 2/3/4`` columns. Those columns are not junk for every row: when a
      message contained a comma, the CSV parser spilled the remainder into them.
      Each continuation field keeps the space that followed the original comma, so
      re-joining with a comma restores the message verbatim. Naively keeping only
      ``v2`` would silently truncate those messages (a real, invisible data bug).
      Stray backslash escape artifacts introduced by the CSV are stripped.
    - Drops exact-duplicate messages **before** any split to avoid train/test
      leakage (the same text landing in both sets would inflate scores).
    """
    raw = pd.read_csv(DATA_PATH, encoding="latin-1")
    text_cols = ["v2"] + [c for c in raw.columns if str(c).startswith("Unnamed")]

    def _join_text(row) -> str:
        parts = [str(row[c]) for c in text_cols if pd.notna(row[c])]
        return ",".join(parts)

    df = pd.DataFrame(
        {
            "label": raw["v1"].map({"ham": HAM, "spam": SPAM}),
            "text": raw.apply(_join_text, axis=1),
        }
    )
    df["text"] = df["text"].str.replace("\\", "", regex=False).str.strip()
    df = df.dropna(subset=["label"])
    df = df[df["text"] != ""]
    if drop_duplicates:
        df = df.drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def clean(
    text: str,
    *,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    stem: bool = False,
    remove_punct: bool = True,
) -> str:
    """Normalise one message. Each preprocessing step is independently toggleable.

    Order: lowercase -> strip punctuation -> tokenize -> drop stopwords -> stem.
    Returns a space-joined string of tokens (ready for a vectorizer).

    The ``remove_punct`` switch is genuine (Phase 3 ablation relies on it):
    - ``remove_punct=True``  strips punctuation, then tokenizes word characters
      (``\\w+``). On this default path the strip already removed all punctuation, so
      the token set is identical to the historical ``\\b\\w+\\b`` tokenizer.
    - ``remove_punct=False`` keeps punctuation as standalone tokens
      (``\\w+|[^\\w\\s]``), so "keep punctuation as features" is a real config.
    """
    if not isinstance(text, str):
        text = str(text)
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = text.translate(_PUNCT_TABLE)
        tokens = _TOKEN_RE.findall(text)
    else:
        tokens = _TOKEN_RE_KEEP_PUNCT.findall(text)
    if remove_stopwords:
        stops = _get_stopwords()
        tokens = [t for t in tokens if t not in stops]
    if stem:
        stemmer = _get_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


class TextCleaner(BaseEstimator, TransformerMixin):
    """``clean`` as a Pipeline step, with the same independent toggles.

    Notebooks 02 and 03 cleaned the text *before* building the pipeline, which was
    fine because every branch of the pipeline wanted cleaned text. Phase 4's hybrid
    feature set does not: the statistical features (uppercase ratio, punctuation
    frequency) must see the **raw** message, because cleaning is exactly what
    destroys that signal. So the pipeline is fed raw text and this transformer
    cleans only the lexical branch.

    Cleaning is a deterministic per-message map with nothing fitted, so running it
    inside the pipeline introduces no leakage; it is equivalent to the earlier
    notebooks' pre-cleaning, which is what the Phase-4 control arm verifies.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        stem: bool = False,
        remove_punct: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.remove_punct = remove_punct

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [
                clean(
                    t,
                    lowercase=self.lowercase,
                    remove_stopwords=self.remove_stopwords,
                    stem=self.stem,
                    remove_punct=self.remove_punct,
                )
                for t in X
            ]
        )


def make_split(df: pd.DataFrame, force: bool = False):
    """Return the ONE stratified 80/20 split as ``(X_train, X_test, y_train, y_test)``.

    The row indices are persisted to ``results/split_indices.npz`` on first call
    and reloaded on every later call, so all notebooks share an identical split.
    Pass ``force=True`` only to regenerate it deliberately.
    """
    if SPLIT_PATH.exists() and not force:
        saved = np.load(SPLIT_PATH)
        train_idx, test_idx = saved["train_idx"], saved["test_idx"]
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=TEST_SIZE,
            stratify=df["label"].values,
            random_state=RANDOM_SEED,
        )
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(SPLIT_PATH, train_idx=train_idx, test_idx=test_idx)

    X = df["text"].values
    y = df["label"].values
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
