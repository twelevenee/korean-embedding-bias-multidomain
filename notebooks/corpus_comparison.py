# %% [markdown]
# # Cross-Corpus Gender Bias Comparison in Korean Word Embeddings
#
# **Research question:** Does Korean news media encode gender-occupational
# associations differently from web/social media and encyclopedic text?
#
# This notebook extends the existing WEAT analysis by adding a third corpus —
# Korean news articles (Naver News) — and comparing bias patterns across three
# distinct institutional registers:
#
# | Corpus | Register | Training data |
# |---|---|---|
# | **Social media** | Informal / user-generated | CC-100 web crawl (FastText) |
# | **Encyclopedic** | Neutral / curated | Wikipedia + Namuwiki (Word2Vec) |
# | **News** | Institutional / journalistic | Naver News articles (Word2Vec, new) |
#
# **Motivation:** News media serves as a formal institutional record that shapes
# public discourse — distinct from how online communities actually use language.
# If news and social media encode bias differently, this has direct implications
# for how AI systems trained on these corpora will behave in practice.

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors, Word2Vec

from corpus_comparison   import run_comparison, CORPUS_LABELS, WEAT_TESTS
from visualize_comparison import plot_grouped_bars, plot_divergence, plot_per_word_heatmap

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.family"] = "DejaVu Sans"

# %% [markdown]
# ## 1. Load models and run WEAT battery
#
# This cell runs the full WEAT battery (3 tests × 3 corpora × 10,000 permutations).
# Expected runtime: ~5 minutes.

# %%
FT_PATH   = PROJECT_ROOT / "models" / "cc.ko.300.bin"
W2V_PATH  = PROJECT_ROOT / "models" / "ko.bin"
NEWS_PATH = PROJECT_ROOT / "models" / "news_w2v.bin"

df = run_comparison(FT_PATH, W2V_PATH, NEWS_PATH, n_perms=10_000)
df.head(9)

# %% [markdown]
# ## 2. Effect sizes at a glance

# %%
pivot = df.pivot(index="test_id", columns="corpus", values="d")[
    ["social_media", "encyclopedic", "news"]
]
pivot.columns = ["Social Media", "Encyclopedic", "News"]
pivot.index   = [t["name"].replace("\n", " ") for t in WEAT_TESTS]
print(pivot.to_string())

# %%
# p-values
pivot_p = df.pivot(index="test_id", columns="corpus", values="p")[
    ["social_media", "encyclopedic", "news"]
]
pivot_p.columns = ["Social Media", "Encyclopedic", "News"]
pivot_p.index   = [t["name"].replace("\n", " ") for t in WEAT_TESTS]
print(pivot_p.to_string())

# %% [markdown]
# ## 3. Grouped bar chart — bias across corpora

# %%
fig = plot_grouped_bars(df, save=True)
plt.show()

# %% [markdown]
# ## 4. News vs. social media divergence
#
# This is the core cross-corpus finding. Positive Δd means the news corpus
# encodes *more* male association than social media for that occupation category.

# %%
fig = plot_divergence(df, save=True)
plt.show()

# %% [markdown]
# ## 5. Per-word association heatmap
#
# Loads all three models into memory simultaneously. Requires ~3 GB RAM.

# %%
print("Loading models for heatmap (this takes ~1-2 min) …")
from gensim.models.fasttext import load_facebook_vectors
models = {
    "social_media": load_facebook_vectors(str(FT_PATH)),
    "encyclopedic": Word2Vec.load(str(W2V_PATH)).wv,
    "news":         KeyedVectors.load_word2vec_format(str(NEWS_PATH),binary=True),
}

fig = plot_per_word_heatmap(models, save=True)
plt.show()

# Free memory
for m in models.values():
    del m

# %% [markdown]
# ## 6. Interpretation
#
# Fill this section after seeing your actual results. Use the template below
# as a guide — replace the bracketed text with what your data shows.

# %%
# ── TEMPLATE — fill in after running ─────────────────────────────────────────
#
# ### Key finding 1 — [corpus] shows highest bias for [occupation category]
#
# The [social_media / news / encyclopedic] corpus shows the strongest
# gender-occupational bias for [T1 / T2 / T3] (d = [X.XX], p = [X.XXX]).
# This suggests that [interpretation].
#
# ### Key finding 2 — News [amplifies / suppresses] expertise bias
#
# For neutral/professional occupations (의사, 교수, 변호사), the news corpus
# shows [higher / lower] male association (d = [X.XX]) than social media
# (d = [X.XX]), Δd = [X.XX].
# This [supports / contradicts] the hypothesis that institutional media
# [reinforces / moderates] the male default for expertise.
#
# ### Key finding 3 — Corpus composition matters
#
# The largest difference between corpora is observed in [T1 / T2 / T3]
# (range: [X.XX] – [X.XX] across corpora), while [T_] shows the most
# consistent bias pattern regardless of register.
# This extends our earlier finding that corpus composition drives bias
# more than model architecture — it now holds across registers too.
#
# ### Limitation
#
# The news Word2Vec model was trained on [N] sentences, which is smaller
# than the CC-100 FastText model. Differences in vocabulary coverage may
# partially account for OOV-filtering effects.

# %% [markdown]
# ## 7. Save results summary

# %%
summary = df[["corpus", "test_id", "d", "p", "significant"]].copy()
summary["corpus_label"] = summary["corpus"].map(CORPUS_LABELS)
summary = summary.sort_values(["test_id", "corpus"])

output_path = PROJECT_ROOT / "results" / "csv" / "corpus_comparison_summary.csv"
summary.to_csv(output_path, index=False)
print(f"Saved → {output_path}")
summary
