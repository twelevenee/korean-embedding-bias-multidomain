# Extension: Cross-Corpus Bias Comparison

> Add this section to the main README.md under a new `## Extension` heading.

---

## Extension — Cross-Corpus Comparison (News vs. Social Media vs. Encyclopedic)

### Motivation

The original analysis found that **corpus composition matters more than model
architecture** — FastText (CC-100 web crawl) shows systematically higher bias
than Word2Vec (Wikipedia/Namuwiki). This raises a natural follow-up question:

> Does **news media** encode gender-occupational bias differently from
> web/social media and encyclopedic text?

News articles represent a distinct institutional register: editorially
gatekept, topic-driven, and written for public consumption. If news corpora
produce different bias patterns than social media, this has direct implications
for how NLP systems trained on these corpora will behave — and for how media
shapes societal representations of gender and work.

### Third Corpus: Korean News (Naver News)

A Word2Vec model (200 dim, window=5, CBOW) was trained on **[N] sentences**
from Naver News articles, sourced via `daekeun-ml/naver-news-summarization-ko`
and tokenized with Okt.

| Model | Corpus | Dim | Vocab |
|---|---|---|---|
| FastText | CC-100 web crawl | 300 | — |
| Word2Vec | Wikipedia + Namuwiki | 200 | — |
| **Word2Vec (new)** | **Naver News** | **200** | **[N]** |

### Results

*(Fill in after running the analysis — see notebooks/corpus_comparison.py)*

| Test | Social Media (d) | Encyclopedic (d) | **News (d)** |
|---|---|---|---|
| T1: Male-coded vs. Female-coded | 1.42* | — | — |
| T2: Neutral/Prof. vs. Female-coded | 1.29* | — | — |
| T3: Neutral/Prof. vs. Male-coded | −0.59 | — | — |

*p < 0.05

**Key findings:**
- *[Fill in after results — e.g., "News shows X% stronger/weaker bias than social media for professional occupations"]*
- *[Fill in divergence finding]*
- *[Fill in any surprising per-word patterns from the heatmap]*

### Figures

| Figure | Description |
|---|---|
| `results/figures/corpus_comparison_bars.png` | Grouped bars: d per test × corpus |
| `results/figures/corpus_divergence.png` | Δd: news − social media per test |
| `results/figures/corpus_heatmap.png` | Per-word association scores × corpus |

### How to Reproduce

```bash
# 1. Install new dependencies
pip install datasets konlpy gensim tqdm

# 2. Download and tokenize news corpus (~45 min)
python scripts/01_acquire_news_corpus.py --limit 150000

# 3. Train Word2Vec on news corpus (~10 min)
python scripts/02_train_news_w2v.py

# 4. Run WEAT comparison across all three corpora (~5 min)
python src/corpus_comparison.py

# 5. Generate visualizations
python src/visualize_comparison.py

# 6. Full narrative analysis
jupyter notebook notebooks/corpus_comparison.py
```

### Limitations

- News Word2Vec trained on [N] sentences vs. the larger CC-100/Wikipedia corpora — 
  vocabulary coverage differences may partially explain OOV effects.
- Static (non-contextual) embeddings only; contextual representations
  (KoBERT, KLUE-RoBERTa) may show different patterns.
- Naver News has its own editorial biases and topic distribution 
  (heavy on politics, economy, sports) that may skew the results.
