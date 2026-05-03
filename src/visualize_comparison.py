"""
visualize_comparison.py — Plots for the multi-corpus WEAT comparison.

Generates three figures saved to results/figures/:
  1. corpus_comparison_bars.png   — grouped bar chart: d per test × corpus
  2. corpus_divergence.png        — news vs. social media delta per test
  3. corpus_heatmap.png           — per-word association score heatmap × corpus

Usage
-----
    from src.visualize_comparison import plot_all
    plot_all(df)          # df from corpus_comparison.run_comparison()

    # or run standalone after generating results/csv/corpus_comparison.csv:
    python src/visualize_comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

FIGURES_DIR = Path("results/figures")

CORPUS_COLORS = {
    "social_media": "#E07B54",
    "encyclopedic": "#5B8DB8",
    "news":         "#4CAF6F",
}
CORPUS_LABELS = {
    "social_media": "Social Media (CC-100)",
    "encyclopedic": "Encyclopedic (Wikipedia)",
    "news":         "News (Naver)",
}

TEST_LABELS = {
    "T1": "Male-coded vs.\nFemale-coded",
    "T2": "Neutral/Prof. vs.\nFemale-coded",
    "T3": "Neutral/Prof. vs.\nMale-coded",
}


# ── Figure 1: grouped bar chart ───────────────────────────────────────────────

def plot_grouped_bars(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Side-by-side bars: one group per WEAT test, one bar per corpus.
    Significant results get a star marker.
    """
    tests   = sorted(df["test_id"].unique())
    corpora = ["social_media", "encyclopedic", "news"]

    n_tests   = len(tests)
    n_corpora = len(corpora)
    bar_width = 0.22
    x = np.arange(n_tests)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")

    for i, corpus in enumerate(corpora):
        subset = df[df["corpus"] == corpus].set_index("test_id")
        heights = [subset.loc[t, "d"] if t in subset.index else 0.0 for t in tests]
        pvals   = [subset.loc[t, "p"] if t in subset.index else 1.0 for t in tests]

        offset = (i - (n_corpora - 1) / 2) * (bar_width + 0.02)
        bars = ax.bar(
            x + offset, heights,
            width=bar_width,
            color=CORPUS_COLORS[corpus],
            label=CORPUS_LABELS[corpus],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        # Add significance stars
        for bar, p in zip(bars, pvals):
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
            if star:
                ypos = bar.get_height() + 0.03 if bar.get_height() >= 0 else bar.get_height() - 0.10
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        star, ha="center", va="bottom", fontsize=10, color="#333333")

    ax.axhline(0, color="#888888", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([TEST_LABELS[t] for t in tests], fontsize=11)
    ax.set_ylabel("WEAT Effect Size (d)", fontsize=11)
    ax.set_title(
        "Gender-Occupational Bias Across Korean Corpora\n"
        "(* p<0.05   ** p<0.01   *** p<0.001)",
        fontsize=12, pad=14,
    )
    ax.legend(framealpha=0.9, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(
        min(-0.3, df["d"].min() - 0.2),
        max(0.3,  df["d"].max() + 0.25),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "corpus_comparison_bars.png", dpi=150, bbox_inches="tight")
        print(f"✓ Saved corpus_comparison_bars.png")
    return fig


# ── Figure 2: divergence plot (news − social media) ───────────────────────────

def plot_divergence(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Shows Δd = news_d − social_media_d for each test.
    Positive = news shows MORE bias; negative = news shows LESS bias.
    """
    tests = sorted(df["test_id"].unique())

    sm   = df[df["corpus"] == "social_media"].set_index("test_id")["d"]
    news = df[df["corpus"] == "news"].set_index("test_id")["d"]

    deltas = [(news.get(t, np.nan) - sm.get(t, np.nan)) for t in tests]
    colors = ["#4CAF6F" if d >= 0 else "#E07B54" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")

    bars = ax.barh(
        [TEST_LABELS[t] for t in tests],
        deltas,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        height=0.45,
    )

    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel("Δd  (News − Social Media)", fontsize=11)
    ax.set_title(
        "Does News Encode Different Bias Than Social Media?\n"
        "Positive = news shows more male association",
        fontsize=12, pad=12,
    )

    for bar, delta in zip(bars, deltas):
        if not np.isnan(delta):
            label_x = delta + (0.01 if delta >= 0 else -0.01)
            ha = "left" if delta >= 0 else "right"
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    f"{delta:+.3f}", va="center", ha=ha, fontsize=10)

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "corpus_divergence.png", dpi=150, bbox_inches="tight")
        print(f"✓ Saved corpus_divergence.png")
    return fig


# ── Figure 3: per-word heatmap across corpora ─────────────────────────────────

def plot_per_word_heatmap(
    models: dict,          # {"social_media": KeyedVectors, ...}
    save: bool = True,
) -> plt.Figure:
    """
    For each occupation word, compute the mean cosine association difference
    s(w, MALE_ATTRS) - s(w, FEMALE_ATTRS) across all three corpora.
    Displayed as a heatmap: words × corpora.
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from word_sets import MALE_ATTRS, FEMALE_ATTRS  # noqa

    all_occupations_raw = []
    try:
        from word_sets import MALE_OCCUPATIONS, FEMALE_OCCUPATIONS, NEUTRAL_OCCUPATIONS
        all_occupations_raw = MALE_OCCUPATIONS + FEMALE_OCCUPATIONS + NEUTRAL_OCCUPATIONS
    except ImportError:
        print("Could not import word_sets — skipping heatmap")
        return None

    corpora  = ["social_media", "encyclopedic", "news"]
    data     = {}

    for corpus, wv in models.items():
        attrs_male   = [w for w in MALE_ATTRS   if w in wv]
        attrs_female = [w for w in FEMALE_ATTRS if w in wv]
        scores = {}
        for word in all_occupations_raw:
            if word not in wv:
                scores[word] = np.nan
                continue
            vec = wv[word]
            sim_m = np.mean([np.dot(vec, wv[a]) / (np.linalg.norm(vec) * np.linalg.norm(wv[a]))
                             for a in attrs_male])
            sim_f = np.mean([np.dot(vec, wv[a]) / (np.linalg.norm(vec) * np.linalg.norm(wv[a]))
                             for a in attrs_female])
            scores[word] = round(float(sim_m - sim_f), 4)
        data[corpus] = scores

    heat_df = pd.DataFrame(data, index=all_occupations_raw)[corpora]
    heat_df.columns = [CORPUS_LABELS[c] for c in corpora]

    fig, ax = plt.subplots(figsize=(9, max(5, len(all_occupations_raw) * 0.42)))
    fig.patch.set_facecolor("white")

    # diverging colormap centred at 0
    vmax = heat_df.abs().max().max()
    im = ax.imshow(
        heat_df.values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax, vmax=vmax,
    )

    ax.set_xticks(range(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns, fontsize=10)
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index, fontsize=10)

    # annotate cells
    for i in range(len(heat_df.index)):
        for j in range(len(heat_df.columns)):
            val = heat_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > vmax * 0.6 else "#222")

    plt.colorbar(im, ax=ax, label="s(w, male) − s(w, female)")
    ax.set_title(
        "Per-word Male Association Scores Across Corpora\n"
        "(Red = male-associated, Blue = female-associated)",
        fontsize=11, pad=12,
    )
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "corpus_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"✓ Saved corpus_heatmap.png")
    return fig


# ── Figure 4: 2×2 framework scatter plot ─────────────────────────────────────

def plot_2x2_framework(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Scatter plot: x = register formality, y = structural gender separation.
    Bubble size = mean |d| across T1/T2. Color = corpus.
    Quadrant labels highlight the theoretical framework.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus_comparison import CORPUS_FRAMEWORK, corpus_color, corpus_label

    # Mean absolute d across T1 and T2 (the "bias magnitude" measure)
    bias = (
        df[df["test_id"].isin(["T1", "T2"])]
        .groupby("corpus")["d"]
        .apply(lambda x: x.abs().mean())
    )

    corpora = [c for c in df["corpus"].unique() if c in CORPUS_FRAMEWORK]
    if not corpora:
        print("No corpora with framework positions — skipping 2×2 plot")
        return None

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    for i, corpus in enumerate(corpora):
        x, y = CORPUS_FRAMEWORK[corpus]
        d_mag = float(bias.get(corpus, 0.1))
        color = corpus_color(corpus, i)
        size = max(80, d_mag * 600)

        ax.scatter(x, y, s=size, color=color, alpha=0.82, zorder=3,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(
            corpus_label(corpus).replace("\n", "\n"),
            (x, y), fontsize=9, ha="center", va="bottom",
            xytext=(0, 10), textcoords="offset points", color="#222",
        )

    # Quadrant lines and labels
    ax.axvline(0.5, color="#ccc", linewidth=1, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#ccc", linewidth=1, linestyle="--", zorder=1)
    quadrant_kw = dict(fontsize=9, color="#aaa", style="italic", ha="center")
    ax.text(0.25, 0.92, "Informal +\nHigh separation", transform=ax.transAxes, **quadrant_kw)
    ax.text(0.75, 0.92, "Formal +\nHigh separation", transform=ax.transAxes, **quadrant_kw)
    ax.text(0.25, 0.05, "Informal +\nLow separation", transform=ax.transAxes, **quadrant_kw)
    ax.text(0.75, 0.05, "Formal +\nLow separation", transform=ax.transAxes, **quadrant_kw)

    ax.set_xlabel("Register Formality  (informal ← → formal)", fontsize=11)
    ax.set_ylabel("Structural Gender Separation  (low ← → high)", fontsize=11)
    ax.set_title(
        "Two-Axis Framework: Register × Structural Separation\n"
        "(bubble size = mean |d| across T1/T2)",
        fontsize=12, pad=14,
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "framework_2x2.png", dpi=150, bbox_inches="tight")
        print("✓ Saved framework_2x2.png")
    return fig


# ── convenience wrapper ───────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, models: dict = None) -> None:
    """Generate all comparison figures from a results DataFrame."""
    plot_grouped_bars(df)
    plot_divergence(df)
    plot_2x2_framework(df)
    if models:
        plot_per_word_heatmap(models)
    print("\n✓ All figures saved to results/figures/")


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = Path("results/csv/corpus_comparison.csv")
    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        print("Run  python src/corpus_comparison.py  first.")
    else:
        df = pd.read_csv(csv_path)
        plot_grouped_bars(df)
        plot_divergence(df)
        plot_2x2_framework(df)
        print("Done. Heatmap requires loaded models — run from notebook.")
