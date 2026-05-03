"""
Step 4 — Time-series WEAT analysis on crime news.

For each year with a corpus file in data/crime_corpus_by_year/,
trains a Word2Vec model and runs the WEAT battery.
Outputs a line plot of d-scores over time.

Requires at least 2 years of data to produce a meaningful plot.
For a single-year corpus, use src/corpus_comparison.py instead.

Usage
-----
    python scripts/04_timeseries_weat.py
    python scripts/04_timeseries_weat.py --min_years 3
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from weat import weat_effect_size, permutation_test   # noqa: E402
from word_sets import (                                # noqa: E402
    MALE_ATTRS, FEMALE_ATTRS,
    MALE_OCCUPATIONS, FEMALE_OCCUPATIONS, NEUTRAL_OCCUPATIONS,
)

BY_YEAR_DIR  = PROJECT_ROOT / "data"  / "crime_corpus_by_year"
MODELS_DIR   = PROJECT_ROOT / "models" / "crime_by_year"
RESULTS_DIR  = PROJECT_ROOT / "results"

WEAT_TESTS = [
    {"id": "T1", "name": "Male vs Female\nOccupations",
     "X": MALE_OCCUPATIONS, "Y": FEMALE_OCCUPATIONS, "A": MALE_ATTRS, "B": FEMALE_ATTRS},
    {"id": "T2", "name": "Neutral/Prof. vs\nFemale Occupations",
     "X": NEUTRAL_OCCUPATIONS, "Y": FEMALE_OCCUPATIONS, "A": MALE_ATTRS, "B": FEMALE_ATTRS},
    {"id": "T3", "name": "Neutral/Prof. vs\nMale Occupations",
     "X": NEUTRAL_OCCUPATIONS, "Y": MALE_OCCUPATIONS, "A": MALE_ATTRS, "B": FEMALE_ATTRS},
]

TEST_COLORS = {"T1": "#E07B54", "T2": "#5B8DB8", "T3": "#4CAF6F"}


class LineCorpus:
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def filter_oov(words, wv):
    return [w for w in words if w in wv]


def train_year_model(corpus_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = LineCorpus(corpus_path)
    model = Word2Vec(
        sentences=corpus,
        vector_size=200,
        window=5,
        min_count=2,
        workers=4,
        sg=0,
        epochs=5,
        seed=42,
    )
    model.wv.save_word2vec_format(str(output_path), binary=True)


def run_weat_for_year(wv, year: int, n_perms: int) -> list[dict]:
    rows = []
    for test in WEAT_TESTS:
        X = filter_oov(test["X"], wv)
        Y = filter_oov(test["Y"], wv)
        A = filter_oov(test["A"], wv)
        B = filter_oov(test["B"], wv)

        if len(X) < 2 or len(Y) < 2 or len(A) < 2 or len(B) < 2:
            rows.append({"year": year, "test_id": test["id"], "d": np.nan, "p": np.nan})
            continue

        d = weat_effect_size(wv, X, Y, A, B)
        p = permutation_test(wv, X, Y, A, B, n_permutations=n_perms)
        rows.append({"year": year, "test_id": test["id"], "d": round(d, 3), "p": round(p, 4)})
        sig = "✓" if p < 0.05 else "✗"
        print(f"    {test['id']}  d={d:+.3f}  p={p:.4f}  {sig}")
    return rows


def plot_timeseries(df: pd.DataFrame) -> None:
    from gensim.models import KeyedVectors

    years = sorted(df["year"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    for test_id, color in TEST_COLORS.items():
        sub = df[df["test_id"] == test_id].set_index("year")["d"].reindex(years)
        ax.plot(years, sub.values, marker="o", label=test_id, color=color, linewidth=2)

        # Mark significant points
        sig = df[(df["test_id"] == test_id) & (df["p"] < 0.05)].set_index("year")["d"]
        for yr, d_val in sig.items():
            ax.scatter(yr, d_val, color=color, s=80, zorder=5)

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xticks(years)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("WEAT Effect Size (d)", fontsize=11)
    ax.set_title(
        "Gender-Occupational Bias in Korean Crime News Over Time\n"
        "(filled markers = p < 0.05)",
        fontsize=12, pad=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = RESULTS_DIR / "figures" / "timeseries_weat.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved → {out}")
    plt.show()


def main(n_perms: int, min_years: int, retrain: bool) -> None:
    year_files = sorted(BY_YEAR_DIR.glob("*.txt"))
    if len(year_files) < min_years:
        print(f"Found {len(year_files)} year(s) in {BY_YEAR_DIR}.")
        print(f"Need at least {min_years} for time-series (--min_years to change).")
        print("Download more years from BIGKinds and re-run 03_acquire_bigkinds_corpus.py.")
        return

    from gensim.models import KeyedVectors

    all_rows = []
    for corpus_path in year_files:
        year = int(corpus_path.stem)
        model_path = MODELS_DIR / f"{year}.bin"

        print(f"\n── {year} ({corpus_path.name}) ──")

        if retrain or not model_path.exists():
            t0 = time.time()
            train_year_model(corpus_path, model_path)
            print(f"  Trained in {time.time()-t0:.1f}s → {model_path}")
        else:
            print(f"  Using cached model → {model_path}")

        wv = KeyedVectors.load_word2vec_format(str(model_path), binary=True)
        rows = run_weat_for_year(wv, year, n_perms)
        all_rows.extend(rows)
        del wv

    df = pd.DataFrame(all_rows)

    csv_out = RESULTS_DIR / "csv" / "timeseries_weat.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"\n✓ Results saved → {csv_out}")
    print(df.pivot(index="year", columns="test_id", values="d").to_string())

    plot_timeseries(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_perms",   type=int,  default=10_000)
    parser.add_argument("--min_years", type=int,  default=2,
                        help="Minimum years required to run (default: 2)")
    parser.add_argument("--retrain",   action="store_true",
                        help="Re-train models even if cached")
    args = parser.parse_args()
    main(args.n_perms, args.min_years, args.retrain)
