"""
word_sets.py — Korean word lists for WEAT gender-occupational bias tests.

Design notes
------------
* Words are chosen as single morphemes so they survive Okt tokenization in
  the news corpus (compound nouns split at morpheme boundaries would appear
  OOV in models trained on Okt-tokenized text).
* Sets are intentionally larger than the WEAT minimum of 4 so that OOV
  filtering still leaves a usable sample in each model.
* Words were selected to match standard WEAT attribute/target categories
  (Caliskan et al., 2017) adapted for Korean social context.
"""

# ── Gender attribute words ────────────────────────────────────────────────────

MALE_ATTRS: list[str] = [
    "남성",   # male (formal)
    "남자",   # man
    "아버지", # father
    "형",     # older brother (male speaker)
    "오빠",   # older brother (female speaker)
    "남편",   # husband
    "소년",   # boy
    "아들",   # son
    "할아버지", # grandfather
]

FEMALE_ATTRS: list[str] = [
    "여성",   # female (formal)
    "여자",   # woman
    "어머니", # mother
    "누나",   # older sister (male speaker)
    "언니",   # older sister (female speaker)
    "아내",   # wife
    "소녀",   # girl
    "딸",     # daughter
    "할머니", # grandmother
]

# ── Occupation target words ───────────────────────────────────────────────────

# Stereotypically male-coded occupations in Korean society
MALE_OCCUPATIONS: list[str] = [
    "군인",   # soldier
    "경찰",   # police officer
    "소방관", # firefighter
    "기술자", # technician / engineer
    "운전기사", # driver
    "과학자", # scientist
    "조종사", # pilot
    "목수",   # carpenter
]

# Stereotypically female-coded occupations in Korean society
# Words chosen for coverage in news corpora (domestic/service terms are OOV in news).
FEMALE_OCCUPATIONS: list[str] = [
    "간호사", # nurse
    "비서",   # secretary
    "교사",   # teacher (primary/secondary — female-skewed in KR)
    "승무원", # flight attendant — appears in aviation news
    "아나운서", # TV announcer/presenter — strongly female-coded in Korea
    "여배우", # actress — female-specific, appears in entertainment news
    "약사",   # pharmacist — female-skewed profession in Korea
    "사서",   # librarian — female-skewed profession in Korea
]

# Professional / expertise occupations — test whether these get male-associated
# (the "male default for expertise" hypothesis).
# Notebook Section 6 specifically calls out 의사, 교수, 변호사.
NEUTRAL_OCCUPATIONS: list[str] = [
    "의사",   # doctor
    "교수",   # professor
    "변호사", # lawyer
    "판사",   # judge
    "회계사", # accountant
    "기자",   # journalist / reporter
    "작가",   # writer / author
    "감독",   # director / coach
]
