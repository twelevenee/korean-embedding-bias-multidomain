"""
Step 2 — Train a Word2Vec model on the news corpus.

Input  : data/news_corpus_sentences.txt  (from 01_acquire_news_corpus.py)
Output : models/news_w2v.bin             (gensim KeyedVectors, binary)

Parameters are chosen to match the existing Kyubyong Word2Vec model:
  - 200 dimensions
  - window = 5
  - min_count = 5
  - CBOW architecture (sg=0)

Runtime : ~5-15 min on a modern laptop for 150 k sentences.

Usage
-----
    python scripts/02_train_news_w2v.py
    python scripts/02_train_news_w2v.py --corpus data/my_corpus.txt --dim 300
"""

import argparse
import logging
import time
from pathlib import Path

from gensim.models import Word2Vec

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── config ────────────────────────────────────────────────────────────────────

DEFAULT_CORPUS = Path("data/news_corpus_sentences.txt")
DEFAULT_OUTPUT = Path("models/news_w2v.bin")


class LineCorpus:
    """Memory-efficient corpus iterator — reads one line at a time."""
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def train(corpus_path: Path, output_path: Path, dim: int, workers: int) -> None:
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run 01_acquire_news_corpus.py first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training Word2Vec on {corpus_path} …")
    print(f"  dim={dim}, window=5, min_count=5, workers={workers}")
    t0 = time.time()

    corpus = LineCorpus(corpus_path)
    model = Word2Vec(
        sentences=corpus,
        vector_size=dim,
        window=5,
        min_count=2,
        workers=workers,
        sg=0,          # CBOW — matches Kyubyong model
        epochs=5,
        seed=42,
    )

    model.wv.save_word2vec_format(str(output_path), binary=True)
    elapsed = time.time() - t0

    vocab_size = len(model.wv)
    print(f"\n✓ Trained in {elapsed:.1f}s")
    print(f"  Vocabulary: {vocab_size:,} words")
    print(f"  Saved → {output_path}")

    # Quick sanity check: nearest neighbours for 의사 (doctor)
    if "의사" in model.wv:
        neighbours = model.wv.most_similar("의사", topn=5)
        print(f"\n  Nearest neighbours for 의사 (doctor):")
        for word, score in neighbours:
            print(f"    {word:12s}  {score:.3f}")
    else:
        print("\n  ⚠ 의사 not in vocabulary — check corpus quality")

    print("\n  Next: python src/corpus_comparison.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",  type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output",  type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dim",     type=int,  default=200,
                        help="Embedding dimension (default: 200, matches Kyubyong W2V)")
    parser.add_argument("--workers", type=int,  default=4)
    args = parser.parse_args()
    train(args.corpus, args.output, args.dim, args.workers)
