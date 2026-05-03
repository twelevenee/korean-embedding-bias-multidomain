"""
weat.py — Word Embedding Association Test (Caliskan et al., 2017).

Implements:
  weat_effect_size(wv, X, Y, A, B) -> float
  permutation_test(wv, X, Y, A, B, n_permutations) -> float (p-value)

All words passed in must already be filtered to vocabulary (use filter_oov first).
"""

import numpy as np
from gensim.models import KeyedVectors


def _assoc(wv: KeyedVectors, word: str, attrs: list[str]) -> float:
    """Mean cosine similarity between word and a set of attribute words."""
    vec = wv[word]
    norms_a = np.array([wv[a] for a in attrs])
    dots = norms_a @ vec
    denom = np.linalg.norm(norms_a, axis=1) * np.linalg.norm(vec)
    return float(np.mean(dots / np.maximum(denom, 1e-10)))


def _s(wv: KeyedVectors, word: str, A: list[str], B: list[str]) -> float:
    """Association score s(w, A, B) = mean_cos(w, A) − mean_cos(w, B)."""
    return _assoc(wv, word, A) - _assoc(wv, word, B)


def weat_effect_size(
    wv: KeyedVectors,
    X: list[str],
    Y: list[str],
    A: list[str],
    B: list[str],
) -> float:
    """
    WEAT effect size d (Caliskan et al., 2017, eq. 1–2).

      d = (mean_{x∈X} s(x,A,B) − mean_{y∈Y} s(y,A,B)) / std_{w∈X∪Y} s(w,A,B)

    Positive d → X is more associated with A (and Y with B).
    """
    scores_X = np.array([_s(wv, w, A, B) for w in X])
    scores_Y = np.array([_s(wv, w, A, B) for w in Y])
    std_all = np.std(np.concatenate([scores_X, scores_Y]))
    if std_all < 1e-10:
        return 0.0
    return float((scores_X.mean() - scores_Y.mean()) / std_all)


def permutation_test(
    wv: KeyedVectors,
    X: list[str],
    Y: list[str],
    A: list[str],
    B: list[str],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> float:
    """
    One-sided permutation test: p(observed S(X,Y,A,B) >= permuted).

    Returns the p-value. Small p → the observed bias is unlikely by chance.
    """
    all_words = X + Y
    n_X = len(X)
    scores = np.array([_s(wv, w, A, B) for w in all_words])
    observed = scores[:n_X].mean() - scores[n_X:].mean()

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(all_words))
        stat = scores[perm[:n_X]].mean() - scores[perm[n_X:]].mean()
        if stat >= observed:
            count += 1
    return count / n_permutations
