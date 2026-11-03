"""Fold-local text and quantity baselines with no cross-gap n-grams."""
import numpy as np
import warnings
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


def word_ngrams(document):
    result = []
    for segment in document:
        result.extend("u:" + t for t in segment)
        result.extend("b:" + a + " " + b for a, b in zip(segment, segment[1:]))
    return result


def char_ngrams(document):
    # Equivalent to char_wb for single whitespace-free tokens; never joins tokens.
    result = []
    for segment in document:
        for token in segment:
            padded = " " + token + " "
            for n in (3, 4, 5):
                if len(padded) < n:
                    result.append(padded)
                else:
                    result.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
                if n >= len(padded):
                    break
    return result


def vectorize(train, test):
    params = dict(min_df=2, sublinear_tf=True, max_features=20000, lowercase=False)
    w = TfidfVectorizer(analyzer=word_ngrams, **params)
    c = TfidfVectorizer(analyzer=char_ngrams, **params)
    try:
        x = hstack([w.fit_transform(train), c.fit_transform(train)], format="csr")
        z = hstack([w.transform(test), c.transform(test)], format="csr")
    except ValueError as e:
        raise ValueError("Fold-local TF-IDF vocabulary is empty or invalid; supply more training text (min_df=2)") from e
    return x, z, w, c


def fit_predict(train, test, arm, seed):
    y = np.array([r.label for r in train])
    if set(y) != {0, 1}:
        raise ValueError("Each task's training fold must contain both classes")
    if arm.startswith("quantity"):
        x = np.log1p([[r.n_tokens] for r in train])
        z = np.log1p([[r.n_tokens] for r in test])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=5000, random_state=seed))
    else:
        x, z, _, _ = vectorize([r.segments for r in train], [r.segments for r in test])
        model = LogisticRegression(C=1, class_weight="balanced", max_iter=5000, random_state=seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            result = model.fit(x, y).predict_proba(z)[:, 1]
    except ConvergenceWarning as e:
        raise ValueError("Logistic regression did not converge; inspect this training partition") from e
    if result.shape != (len(test),) or not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Invalid predicted probabilities")
    return result
