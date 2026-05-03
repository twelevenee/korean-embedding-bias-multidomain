"""
corpus_comparison.py — Multi-corpus WEAT comparison.

Fixed corpora (always loaded):
  social_media : FastText CC-100 web crawl
  encyclopedic : Word2Vec Wikipedia/Namuwiki
  news         : Word2Vec general news (Naver/KLUE)

Optional domain corpora (loaded when model file exists):
  crime, politics, sports, entertainment, …   (BIGKinds)
  Add any domain with --extra name:models/name_w2v.bin

Usage (from project root)
--------------------------
    python src/corpus_comparison.py
    python src/corpus_comparison.py \\
        --extra crime:models/crime_w2v.bin \\
        --extra politics:models/politics_w2v.bin \\
        --extra sports:models/sports_w2v.bin \\
        --extra entertainment:models/entertainment_w2v.bin
"""

import argparse
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from weat import weat_effect_size, permutation_test   # noqa: E402
from word_sets import (                                # noqa: E402
    MALE_ATTRS, FEMALE_ATTRS,
    MALE_OCCUPATIONS, FEMALE_OCCUPATIONS, NEUTRAL_OCCUPATIONS,
)


# ── default model paths ───────────────────────────────────────────────────────
DEFAULT_FT_PATH   = PROJECT_ROOT / "models" / "cc.ko.300.bin"
DEFAULT_W2V_PATH  = PROJECT_ROOT / "models" / "ko.bin"
DEFAULT_NEWS_PATH = PROJECT_ROOT / "models" / "news_w2v.bin"
RESULTS_DIR       = PROJECT_ROOT / "results"

# Domains auto-detected from models/ directory (name must match *_w2v.bin)
AUTO_DOMAINS = ["crime", "politics", "sports", "entertainment"]


# ── corpus metadata ───────────────────────────────────────────────────────────
# Labels and colors for known corpora; unknown domains get defaults.
CORPUS_LABELS = {
    "social_media":   "Social Media\n(CC-100)",
    "encyclopedic":   "Encyclopedic\n(Wikipedia)",
    "news":           "News\n(Naver)",
    "crime":          "Crime\n(BIGKinds)",
    "politics":       "Politics\n(BIGKinds)",
    "sports":         "Sports\n(BIGKinds)",
    "entertainment":  "Entertainment\n(BIGKinds)",
}

CORPUS_COLORS = {
    "social_media":   "#E07B54",
    "encyclopedic":   "#5B8DB8",
    "news":           "#4CAF6F",
    "crime":          "#9B59B6",
    "politics":       "#E74C3C",
    "sports":         "#F39C12",
    "entertainment":  "#1ABC9C",
}

# 2×2 framework positions for scatter plot
# (register_formality, structural_gender_separation)  both 0–1
CORPUS_FRAMEWORK = {
    "social_media":   (0.1, 0.5),   # informal, moderate separation
    "encyclopedic":   (0.9, 0.2),   # formal, low separation
    "news":           (0.8, 0.3),   # formal, low-moderate
    "crime":          (0.7, 0.6),   # formal, moderate-high (victim/perp framing)
    "politics":       (0.8, 0.7),   # formal, high (institutional under-representation)
    "sports":         (0.5, 0.9),   # semi-formal, very high (physical separation)
    "entertainment":  (0.3, 0.4),   # informal-ish, different structure
}


# ── WEAT tests ────────────────────────────────────────────────────────────────
WEAT_TESTS = [
    {
        "id":   "T1",
        "name": "Male-coded vs Female-coded\nOccupations",
        "X":    MALE_OCCUPATIONS,
        "Y":    FEMALE_OCCUPATIONS,
        "A":    MALE_ATTRS,
        "B":    FEMALE_ATTRS,
    },
    {
        "id":   "T2",
        "name": "Neutral/Professional vs Female-coded\nOccupations",
        "X":    NEUTRAL_OCCUPATIONS,
        "Y":    FEMALE_OCCUPATIONS,
        "A":    MALE_ATTRS,
        "B":    FEMALE_ATTRS,
    },
    {
        "id":   "T3",
        "name": "Neutral/Professional vs Male-coded\nOccupations",
        "X":    NEUTRAL_OCCUPATIONS,
        "Y":    MALE_OCCUPATIONS,
        "A":    MALE_ATTRS,
        "B":    FEMALE_ATTRS,
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def filter_oov(words: list[str], wv: KeyedVectors) -> list[str]:
    return [w for w in words if w in wv]


def corpus_label(name: str) -> str:
    return CORPUS_LABELS.get(name, name.replace("_", " ").title())


def corpus_color(name: str, idx: int = 0) -> str:
    fallback = ["#7F8C8D", "#BDC3C7", "#95A5A6"]
    return CORPUS_COLORS.get(name, fallback[idx % len(fallback)])


def run_weat_battery(
    wv: KeyedVectors,
    corpus_name: str,
    n_permutations: int = 10_000,
) -> list[dict]:
    rows = []
    for test in WEAT_TESTS:
        X = filter_oov(test["X"], wv)
        Y = filter_oov(test["Y"], wv)
        A = filter_oov(test["A"], wv)
        B = filter_oov(test["B"], wv)

        if len(X) < 2 or len(Y) < 2 or len(A) < 2 or len(B) < 2:
            print(f"  ⚠  {corpus_name}/{test['id']}: too many OOV — skipping")
            rows.append({
                "corpus": corpus_name, "test_id": test["id"],
                "test_name": test["name"],
                "d": np.nan, "p": np.nan,
                "n_X": len(X), "n_Y": len(Y), "significant": False,
            })
            continue

        d = weat_effect_size(wv, X, Y, A, B)
        p = permutation_test(wv, X, Y, A, B, n_permutations=n_permutations)
        rows.append({
            "corpus": corpus_name, "test_id": test["id"],
            "test_name": test["name"],
            "d": round(d, 3), "p": round(p, 4),
            "n_X": len(X), "n_Y": len(Y),
            "significant": p < 0.05,
        })
        print(f"  {test['id']}  d={d:+.3f}  p={p:.4f}  {'✓' if p < 0.05 else '✗'}")
    return rows


def load_model(path: Path, fmt: str = "word2vec") -> KeyedVectors:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    if fmt == "fasttext":
        return load_facebook_vectors(str(path))
    if fmt == "gensim":
        with open(str(path), "rb") as f:
            model = pickle.load(f, encoding="latin1")
        if hasattr(model, "syn0"):
            kv = KeyedVectors(vector_size=model.layer1_size)
            kv.add_vectors(list(model.vocab.keys()), model.syn0)
            return kv
        return model.wv if hasattr(model, "wv") else model
    return KeyedVectors.load_word2vec_format(str(path), binary=True)


# ── main ──────────────────────────────────────────────────────────────────────

def run_comparison(
    ft_path:   Path,
    w2v_path:  Path,
    news_path: Path,
    extra:     dict[str, Path] | None = None,
    n_perms:   int = 10_000,
) -> pd.DataFrame:
    """
    extra: {corpus_name: model_path} for any additional domain corpora.
           Models that don't exist on disk are silently skipped.
    """
    models: dict[str, tuple[Path, str]] = {
        "social_media": (ft_path,   "fasttext"),
        "encyclopedic": (w2v_path,  "gensim"),
        "news":         (news_path, "word2vec"),
    }
    for name, path in (extra or {}).items():
        if path.exists():
            models[name] = (path, "word2vec")
        else:
            print(f"  ⚠  Skipping '{name}': {path} not found")

    all_rows = []
    for corpus_name, (path, fmt) in models.items():
        print(f"\n── {corpus_label(corpus_name)} ──")
        wv = load_model(path, fmt=fmt)
        all_rows.extend(run_weat_battery(wv, corpus_name, n_permutations=n_perms))
        del wv

    df = pd.DataFrame(all_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "csv" / "corpus_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved → {csv_path}")

    pivot = df.pivot(index="corpus", columns="test_id", values="d")
    print("\nEffect sizes (d) by corpus:")
    print(pivot.to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_path",   type=Path, default=DEFAULT_FT_PATH)
    parser.add_argument("--w2v_path",  type=Path, default=DEFAULT_W2V_PATH)
    parser.add_argument("--news_path", type=Path, default=DEFAULT_NEWS_PATH)
    parser.add_argument("--n_perms",   type=int,  default=10_000)
    parser.add_argument(
        "--extra", action="append", default=[],
        metavar="NAME:PATH",
        help="Extra domain corpus: --extra crime:models/crime_w2v.bin  (repeatable)"
    )
    args = parser.parse_args()

    # Auto-detect known domain models even without --extra
    extra: dict[str, Path] = {}
    for domain in AUTO_DOMAINS:
        p = PROJECT_ROOT / "models" / f"{domain}_w2v.bin"
        if p.exists():
            extra[domain] = p

    # Parse explicit --extra name:path overrides
    for item in args.extra:
        name, _, path_str = item.partition(":")
        extra[name] = Path(path_str)

    df = run_comparison(
        args.ft_path, args.w2v_path, args.news_path,
        extra=extra, n_perms=args.n_perms,
    )
    print("\nDone. Run src/visualize_comparison.py for figures.")
