# How to add this extension to your repo

## Step 1 — Copy the files

From this folder, copy into your existing `korean-embedding-bias/` clone:

```
scripts/01_acquire_news_corpus.py  →  korean-embedding-bias/scripts/
scripts/02_train_news_w2v.py       →  korean-embedding-bias/scripts/
src/corpus_comparison.py           →  korean-embedding-bias/src/
src/visualize_comparison.py        →  korean-embedding-bias/src/
notebooks/corpus_comparison.py     →  korean-embedding-bias/notebooks/
README_ADDITION.md                 →  (merge into README.md)
```

## Step 2 — Install new dependencies

```bash
pip install datasets konlpy gensim tqdm
```

> KoNLPy requires Java. If you don't have it:
> macOS:  brew install openjdk
> Ubuntu: sudo apt install default-jdk

## Step 3 — Run the pipeline

```bash
cd korean-embedding-bias

# ~45-60 min — downloads 150k news sentences and tokenizes with Okt
python scripts/01_acquire_news_corpus.py --limit 150000

# ~5-15 min — trains Word2Vec on news corpus
python scripts/02_train_news_w2v.py

# ~5 min — runs WEAT on all 3 corpora, saves results/csv/corpus_comparison.csv
python src/corpus_comparison.py

# instant — generates 2 of the 3 figures
python src/visualize_comparison.py

# full narrative analysis — fill in the Interpretation section after running
jupyter notebook notebooks/corpus_comparison.py
```

## Step 4 — Fill in your findings

After running, open `notebooks/corpus_comparison.py` and fill in Section 6
(Interpretation) with your actual results. This becomes the written analysis
you can reference in your application and eventually a paper section.

## Step 5 — Update the CV bullet

Replace the existing WEAT bullet with something like:

> "Extended WEAT analysis to compare gender-occupational bias across three 
> Korean corpus types (web/social media, encyclopedic, news); found [finding], 
> suggesting news media [amplifies/moderates] the male-default-for-expertise 
> pattern differently than online discourse."

## Notes on timing

- The bottleneck is `01_acquire_news_corpus.py` — Okt tokenization is slow.
  If you need to finish faster, use `--limit 50000` (faster but smaller vocab).
- The Word2Vec training itself takes ~5 min regardless.
- `corpus_comparison.py` with 10,000 permutations takes ~5 min per corpus.
  Use `--n_perms 1000` for a quick first pass.
