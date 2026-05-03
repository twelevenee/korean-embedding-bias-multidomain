"""
neighbor_analysis.py — Word-level nearest neighbor analysis.

For each focus word, shows:
  1. Gender association score  s(w, MALE_ATTRS) - s(w, FEMALE_ATTRS)  across corpora
  2. Top-k nearest neighbors per corpus, color-coded by their own gender association

This explains *why* domain WEAT scores came out the way they did.

Focus words are chosen to cover:
  - Neutral/professional occupations (T2 test — does "expert" = male?)
  - Female-coded occupations  (T1 test — how female-associated are they, really?)
  - The words that drove Entertainment T1 significance

Usage
-----
    python src/neighbor_analysis.py
    python src/neighbor_analysis.py --topk 8 --focus 의사 감독 간호사 여배우
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pickle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from word_sets import MALE_ATTRS, FEMALE_ATTRS  # noqa: E402

RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"

# Corpora to load (order = display order)
CORPUS_SPECS = [
    ("social_media",  PROJECT_ROOT / "models" / "cc.ko.300.bin",      "fasttext"),
    ("entertainment", PROJECT_ROOT / "models" / "entertainment_w2v.bin", "word2vec"),
    ("sports",        PROJECT_ROOT / "models" / "sports_w2v.bin",        "word2vec"),
    ("crime",         PROJECT_ROOT / "models" / "crime_w2v.bin",         "word2vec"),
    ("politics",      PROJECT_ROOT / "models" / "politics_w2v.bin",      "word2vec"),
    ("news",          PROJECT_ROOT / "models" / "news_w2v.bin",          "word2vec"),
    ("encyclopedic",  PROJECT_ROOT / "models" / "ko.bin",               "gensim"),
]

CORPUS_COLORS = {
    "social_media":   "#E07B54",
    "encyclopedic":   "#5B8DB8",
    "news":           "#4CAF6F",
    "crime":          "#9B59B6",
    "politics":       "#E74C3C",
    "sports":         "#F39C12",
    "entertainment":  "#1ABC9C",
}

# Focus words — chosen to explain T1/T2 patterns
DEFAULT_FOCUS = [
    "의사",    # neutral/expert — T2: does sports/entertainment frame this as male?
    "감독",    # director/coach — used totally differently in sports vs entertainment
    "교수",    # professor — neutral expert
    "간호사",  # nurse — core female occupation
    "여배우",  # actress — drove entertainment T1
    "아나운서", # announcer — drove entertainment T1
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(path: Path, fmt: str) -> KeyedVectors | None:
    if not path.exists():
        return None
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


def gender_score(word: str, wv: KeyedVectors) -> float | None:
    """s(w, MALE_ATTRS) - s(w, FEMALE_ATTRS). None if OOV."""
    if word not in wv:
        return None
    male_a  = [a for a in MALE_ATTRS  if a in wv]
    female_a = [a for a in FEMALE_ATTRS if a in wv]
    if not male_a or not female_a:
        return None
    vec = wv[word]
    sim_m = np.mean([np.dot(vec, wv[a]) / (np.linalg.norm(vec) * np.linalg.norm(wv[a]))
                     for a in male_a])
    sim_f = np.mean([np.dot(vec, wv[a]) / (np.linalg.norm(vec) * np.linalg.norm(wv[a]))
                     for a in female_a])
    return float(sim_m - sim_f)


def neighbor_scores(word: str, wv: KeyedVectors, topk: int) -> list[tuple[str, float, float]]:
    """Return [(neighbor, cosine_sim, gender_score), …]."""
    if word not in wv:
        return []
    neighbors = wv.most_similar(word, topn=topk)
    result = []
    for nbr, cos in neighbors:
        gs = gender_score(nbr, wv)
        result.append((nbr, round(cos, 3), round(gs, 3) if gs is not None else 0.0))
    return result


# ── Figure 1: association score heatmap across corpora ───────────────────────

def plot_association_heatmap(
    focus_words: list[str],
    models: dict[str, KeyedVectors],
    save: bool = True,
) -> plt.Figure:
    """
    Rows = focus words, columns = corpora.
    Cell value = s(w, male) - s(w, female).
    Red = male-associated, blue = female-associated.
    """
    corpus_names = list(models.keys())
    scores = {}
    for corpus, wv in models.items():
        scores[corpus] = [gender_score(w, wv) for w in focus_words]

    data = np.array([[scores[c][i] if scores[c][i] is not None else np.nan
                      for c in corpus_names]
                     for i in range(len(focus_words))])

    fig, ax = plt.subplots(figsize=(len(corpus_names) * 1.6, len(focus_words) * 0.7 + 1.2))
    fig.patch.set_facecolor("white")

    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(corpus_names)))
    ax.set_xticklabels(corpus_names, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(focus_words)))
    ax.set_yticklabels(focus_words, fontsize=11)

    for i in range(len(focus_words)):
        for j in range(len(corpus_names)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.55 else "#222"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
            else:
                ax.text(j, i, "OOV", ha="center", va="center",
                        fontsize=8, color="#aaa")

    plt.colorbar(im, ax=ax, label="s(w, male) − s(w, female)", shrink=0.7)
    ax.set_title(
        "Gender Association of Focus Words Across Corpora\n"
        "(Red = male-associated, Blue = female-associated)",
        fontsize=12, pad=12,
    )
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "neighbor_association_heatmap.png",
                    dpi=150, bbox_inches="tight")
        print("✓ Saved neighbor_association_heatmap.png")
    return fig


# ── Figure 2: nearest neighbor grid for a single focus word ──────────────────

def plot_neighbor_grid(
    word: str,
    models: dict[str, KeyedVectors],
    topk: int = 6,
    save: bool = True,
) -> plt.Figure:
    """
    One panel per corpus. Each panel shows top-k neighbors as colored bars
    (bar length = cosine similarity, color = neighbor's gender association).
    """
    corpus_names = list(models.keys())
    n = len(corpus_names)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.8 + 0.8))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten()

    for idx, (corpus, wv) in enumerate(models.items()):
        ax = axes_flat[idx]
        nbrs = neighbor_scores(word, wv, topk)

        if not nbrs:
            ax.text(0.5, 0.5, "OOV", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa", fontsize=12)
            ax.set_title(corpus, fontsize=9, color=CORPUS_COLORS.get(corpus, "#555"))
            ax.axis("off")
            continue

        words_  = [n[0] for n in nbrs][::-1]
        cosines = [n[1] for n in nbrs][::-1]
        gscores = [n[2] for n in nbrs][::-1]

        # Color by gender score: red=male, blue=female, grey=neutral
        cmap = plt.cm.RdBu_r
        vmax = max(abs(g) for g in gscores) or 0.1
        bar_colors = [cmap(0.5 + g / (2 * vmax)) for g in gscores]

        bars = ax.barh(range(len(words_)), cosines, color=bar_colors,
                       edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(words_)))
        ax.set_yticklabels(words_, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("cosine sim", fontsize=8)
        ax.set_title(corpus, fontsize=9, color=CORPUS_COLORS.get(corpus, "#555"),
                     fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)

    # Hide unused panels
    for idx in range(len(corpus_names), len(axes_flat)):
        axes_flat[idx].axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(color=plt.cm.RdBu_r(0.85), label="Male-associated neighbor"),
        mpatches.Patch(color=plt.cm.RdBu_r(0.5),  label="Neutral neighbor"),
        mpatches.Patch(color=plt.cm.RdBu_r(0.15), label="Female-associated neighbor"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'Nearest Neighbors of  "{word}"  by Corpus\n'
        f"(bar color = neighbor's gender association)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = FIGURES_DIR / f"neighbors_{word}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"✓ Saved neighbors_{word}.png")
    return fig


# ── CSV export ────────────────────────────────────────────────────────────────

def export_neighbor_table(
    focus_words: list[str],
    models: dict[str, KeyedVectors],
    topk: int,
) -> pd.DataFrame:
    rows = []
    for word in focus_words:
        for corpus, wv in models.items():
            nbrs = neighbor_scores(word, wv, topk)
            for rank, (nbr, cos, gs) in enumerate(nbrs, 1):
                rows.append({
                    "focus_word": word,
                    "corpus": corpus,
                    "rank": rank,
                    "neighbor": nbr,
                    "cosine": cos,
                    "gender_score": gs,
                })
            if not nbrs:
                rows.append({
                    "focus_word": word, "corpus": corpus,
                    "rank": 0, "neighbor": "OOV",
                    "cosine": np.nan, "gender_score": np.nan,
                })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "csv" / "neighbor_analysis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✓ Saved {out}")
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main(focus_words: list[str], topk: int) -> None:
    # Load available models
    models: dict[str, KeyedVectors] = {}
    for name, path, fmt in CORPUS_SPECS:
        wv = load_model(path, fmt)
        if wv is not None:
            models[name] = wv
            print(f"  Loaded {name}")
        else:
            print(f"  Skipped {name} (model not found)")

    if not models:
        print("No models found.")
        return

    print(f"\nAnalyzing {len(focus_words)} focus words across {len(models)} corpora …\n")

    # Figure 1: association heatmap
    plot_association_heatmap(focus_words, models)

    # Figure 2: neighbor grid for each focus word
    for word in focus_words:
        in_any = any(word in wv for wv in models.values())
        if in_any:
            plot_neighbor_grid(word, models, topk=topk)
        else:
            print(f"  ⚠  '{word}' OOV in all corpora — skipping grid")

    # CSV
    export_neighbor_table(focus_words, models, topk)

    print(f"\n✓ All figures → {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus", nargs="+", default=DEFAULT_FOCUS,
                        help="Focus words to analyze (default: %(default)s)")
    parser.add_argument("--topk",  type=int, default=6,
                        help="Number of nearest neighbors per corpus (default: 6)")
    args = parser.parse_args()
    main(args.focus, args.topk)
