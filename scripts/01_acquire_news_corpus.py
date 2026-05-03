"""
Step 1 — Download Korean news corpus and pre-tokenize with Okt.

Sources (all Korean news register)
-------
  1. daekeun-ml/naver-news-summarization-ko  (~14 k sentences, full articles)
  2. klue/mrc                                (~24 k passages, 60 % from 연합뉴스)
  3. klue/ynat                               (~45 k news headlines)

Output : data/news_corpus_sentences.txt
         One line = one Okt-tokenized sentence (space-separated morphemes).
         This format is consumed directly by train_news_w2v.py.

Expected sentence count : ~100 k+
Runtime  : ~30-50 min (Okt is the bottleneck)
Requires : pip install datasets konlpy tqdm

Usage
-----
    python scripts/01_acquire_news_corpus.py
"""

import re
from pathlib import Path

from datasets import load_dataset
from konlpy.tag import Okt
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

OUTPUT_PATH   = Path("data/news_corpus_sentences.txt")
MIN_SENT_LEN  = 5   # minimum morpheme tokens — low enough to keep short sentences

_SENT_RE = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=죠\.)\s+")


def split_sentences(text: str) -> list[str]:
    sents = _SENT_RE.split(text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 5]


def tokenize(okt: Okt, sentence: str) -> list[str]:
    return [m for m in okt.morphs(sentence, stem=False) if len(m) > 1]


def process_texts(fout, okt: Okt, texts: list[str], pbar: tqdm) -> int:
    n = 0
    for text in texts:
        for sent in split_sentences(text):
            tokens = tokenize(okt, sent)
            if len(tokens) < MIN_SENT_LEN:
                continue
            fout.write(" ".join(tokens) + "\n")
            n += 1
            pbar.update(1)
    return n


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    okt = Okt()
    n_total = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        # ── Source 1: Naver News full articles ────────────────────────────────
        print("[1/3] daekeun-ml/naver-news-summarization-ko …")
        ds = load_dataset(
            "daekeun-ml/naver-news-summarization-ko",
            split="train",
            streaming=True,
        )
        with tqdm(desc="  Naver articles") as pbar:
            for item in ds:
                text = (item.get("title", "") + " " + item.get("content", "")).strip()
                if text:
                    n_total += process_texts(fout, okt, [text], pbar)
        print(f"  → {n_total:,} sentences so far")

        # ── Source 2: KLUE MRC — long news/wiki passages ──────────────────────
        # guid prefix tells us the source: "klue-mrc-v1_train_news_*" = 연합뉴스
        # We take all contexts (news + wiki) since the register is formal Korean.
        print("\n[2/3] klue/mrc contexts …")
        ds_mrc = load_dataset("klue", "mrc", split="train+validation")
        contexts = list({item["context"] for item in ds_mrc if item.get("context")})
        before = n_total
        with tqdm(total=len(contexts), desc="  MRC passages") as pbar:
            n_total += process_texts(fout, okt, contexts, pbar)
        print(f"  → {n_total - before:,} new sentences ({n_total:,} total)")

        # ── Source 3: KLUE YNAT headlines ─────────────────────────────────────
        print("\n[3/3] klue/ynat headlines …")
        ds_ynat = load_dataset("klue", "ynat", split="train+validation")
        headlines = [item["title"] for item in ds_ynat if item.get("title")]
        before = n_total
        with tqdm(total=len(headlines), desc="  YNAT headlines") as pbar:
            n_total += process_texts(fout, okt, headlines, pbar)
        print(f"  → {n_total - before:,} new sentences ({n_total:,} total)")

    print(f"\n✓ Wrote {n_total:,} sentences → {OUTPUT_PATH}")
    print("  Next: python scripts/02_train_news_w2v.py")


if __name__ == "__main__":
    main()
