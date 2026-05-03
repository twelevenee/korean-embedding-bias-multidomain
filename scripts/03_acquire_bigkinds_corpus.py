"""
Step 3 — Tokenize a BIGKinds domain corpus.

Reusable for any domain downloaded from BIGKinds (crime, politics, sports, entertainment, …).

Input  : data/NewsResult_<domain>_*.xlsx   (BIGKinds export)
Output : data/<name>_corpus_sentences.txt           (full corpus)
         data/<name>_corpus_by_year/<YYYY>.txt      (per-year, for time-series)

BIGKinds column mapping
-----------------------
  일자          : date (int YYYYMMDD)
  제목          : title
  본문          : body
  분석제외 여부  : exclude flag (1 = skip)

Usage
-----
    # crime (default)
    python scripts/03_acquire_bigkinds_corpus.py

    # politics
    python scripts/03_acquire_bigkinds_corpus.py \\
        --input "data/NewsResult_politics_*.xlsx" --name politics

    # sports
    python scripts/03_acquire_bigkinds_corpus.py \\
        --input "data/NewsResult_sports_*.xlsx" --name sports

    # entertainment
    python scripts/03_acquire_bigkinds_corpus.py \\
        --input "data/NewsResult_entertainment_*.xlsx" --name entertainment
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm

MIN_SENT_LEN = 5

_SENT_RE = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=요\.)\s+|(?<=죠\.)\s+")


def load_bigkinds(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_excel(p, dtype={"일자": str}) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    if "분석제외 여부" in df.columns:
        df = df[df["분석제외 여부"].isna() | (df["분석제외 여부"] == 0)]

    df["date"] = pd.to_datetime(df["일자"], format="%Y%m%d", errors="coerce")
    df["year"] = df["date"].dt.year
    return df.dropna(subset=["date"]).reset_index(drop=True)


def split_sentences(text: str) -> list[str]:
    sents = _SENT_RE.split(str(text).strip())
    return [s.strip() for s in sents if len(s.strip()) > 5]


def tokenize(okt: Okt, sentence: str) -> list[str]:
    return [m for m in okt.morphs(sentence, stem=False) if len(m) > 1]


def main(input_glob: str, name: str) -> None:
    output_path = Path(f"data/{name}_corpus_sentences.txt")
    by_year_dir = Path(f"data/{name}_corpus_by_year")

    paths = sorted(Path(".").glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    print(f"[{name}] Found {len(paths)} file(s): {[p.name for p in paths]}")

    df = load_bigkinds(paths)
    print(f"  Articles after filtering: {len(df):,}")
    print(f"  Years: {sorted(df['year'].dropna().unique().astype(int).tolist())}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_year_dir.mkdir(parents=True, exist_ok=True)

    okt = Okt()
    year_buffers: dict[int, list[str]] = defaultdict(list)
    n_total = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Tokenizing {name}"):
            text = " ".join([
                str(row.get("제목", "") or ""),
                str(row.get("본문", "")  or ""),
            ]).strip()
            if not text:
                continue

            year = int(row["year"]) if pd.notna(row["year"]) else None

            for sent in split_sentences(text):
                tokens = tokenize(okt, sent)
                if len(tokens) < MIN_SENT_LEN:
                    continue
                line = " ".join(tokens) + "\n"
                fout.write(line)
                if year:
                    year_buffers[year].append(line)
                n_total += 1

    for year, lines in sorted(year_buffers.items()):
        year_path = by_year_dir / f"{year}.txt"
        with open(year_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  {year}: {len(lines):,} sentences → {year_path}")

    print(f"\n✓ [{name}] {n_total:,} sentences → {output_path}")
    print(f"  Next: python scripts/02_train_news_w2v.py "
          f"--corpus {output_path} --output models/{name}_w2v.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",  default="crime",
                        help="Domain name, e.g. crime / politics / sports / entertainment")
    parser.add_argument("--input", default=None,
                        help="Glob pattern for BIGKinds Excel file(s). "
                             "Defaults to data/NewsResult_*_<name>.xlsx")
    args = parser.parse_args()
    glob = args.input or f"data/NewsResult_*_{args.name}.xlsx"
    main(glob, args.name)
