"""
07b_nmf_content_filter.py

Second-pass content filter using Non-negative Matrix Factorization (NMF).

The title-based is_news filter (step 04) catches articles with obvious
non-news titles (e.g., "OBITUARIES", "DEATH NOTICES"), but misses articles
whose titles are neutral while the body text is clearly non-news content.
NMF topic modeling on the same TF feature matrices used for slant estimation
reveals that ~10% of articles passing the title filter are obituary-like
content and ~1% are Spanish-language articles.

Simple keyword/rule-based filters cannot replicate NMF's detection: validation
shows only 38% recall for obituaries and 5% for Spanish (see robustness/
validate_rulebased_vs_nmf.py), because NMF captures latent co-occurrence
patterns (e.g., "home" + "mr" + "st" + "service" appearing together) that
individual keyword matching cannot.

Method:
  1. Fit 10-topic NMF on 500K random subsample of news articles
  2. Identify obituary and Spanish topics by top-word matching
  3. For each congress, assign dominant topic to every news article
  4. Save per-congress boolean exclusion masks

Inputs:
  - newspapers/07_newspaper_features_cong_{cong}.npz  (from INPUT_NEWS_DIR)
  - newspapers/04_newspaper_labeled_cong_{cong}.parquet
  - models/06_shared_vocab_mask.npy
  - speeches/05_vectorizer.joblib

Outputs:
  - models/07b_nmf_model.joblib
  - models/07b_nmf_topic_labels.json
  - newspapers/07b_nmf_exclude_cong_{cong}.npy  (boolean, per article)
"""

import gc
import json
import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import NMF
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import pipeline_config as cfg

# ── Paths ─────────────────────────────────────────────────────────
LABEL_DIR = cfg.NEWSPAPER_LABELS
INPUT_NEWS_DIR = cfg.INPUT_NEWS_DIR
INPUT_SPEECH_DIR = cfg.INPUT_SPEECH_DIR
MODEL_DIR = cfg.MODEL_DIR
OUT_DIR = cfg.NEWS_DIR

# ── NMF settings ─────────────────────────────────────────────────
N_TOPICS = 10
RANDOM_STATE = 42
SUBSAMPLE_SIZE = 500_000
MAX_ITER = 300

# ── Topic identification keywords ────────────────────────────────
OBITUARY_KWS = [
    "die", "died", "funeral", "church", "born", "son", "daughter",
    "wife", "husband", "surviv", "servic", "cemeteri", "burial",
]
SPANISH_KWS = [
    "el", "en", "lo", "su", "la", "de", "por", "para", "con",
    "una", "los", "las", "del", "que", "como",
]

# ── Topic labeling (for documentation) ───────────────────────────
TOPIC_CATEGORIES = {
    "Sports": ["game", "season", "team", "play", "player", "coach", "win",
               "yard", "goal", "score", "hit", "run", "leagu", "basket"],
    "Obituaries": ["die", "funer", "church", "born", "son", "daughter",
                   "wife", "husband", "surviv", "cemeteri", "burial",
                   "pallbear", "pastor", "baptist"],
    "Crime & police": ["polic", "arrest", "charg", "suspect", "crime",
                       "investig", "offic", "prison", "sentenc", "victim",
                       "shoot", "robber", "drug", "murder"],
    "Local government": ["citi", "council", "commiss", "board", "meet",
                         "plan", "propos", "vote", "approv", "budget",
                         "develop", "project", "resid", "propert"],
    "County & judicial": ["court", "case", "judg", "attorney", "lawyer",
                          "trial", "lawsuit", "plaintiff", "defend",
                          "jurisdict", "clerk", "counti"],
    "National & business": ["compani", "million", "billion", "market",
                            "stock", "profit", "industri", "corpor",
                            "busi", "trade", "invest", "bank"],
    "Business & economy": ["percent", "rate", "tax", "cost", "price",
                           "pay", "job", "worker", "employ", "econom",
                           "incom", "increas"],
    "Arts & culture": ["music", "film", "book", "art", "show", "theater",
                       "perform", "museum", "festiv", "concert", "award",
                       "cultur", "direct"],
    "Spanish-language": ["el", "en", "lo", "su", "se", "le", "contra",
                         "pie", "cuba", "gobierno", "nacion"],
    "Government & institutions": ["state", "senat", "depart", "govern",
                                   "feder", "agenc", "univers", "colleg"],
    "General features": ["time", "day", "week", "long", "thing", "way",
                         "life", "work", "look", "make", "come", "back",
                         "feel", "call"],
    "Human interest": ["children", "famili", "parent", "student", "teacher",
                       "kid", "girl", "boy", "mother", "father", "age",
                       "woman", "man", "young"],
    "Politics & elections": ["democrat", "republican", "vote", "elect",
                             "campaign", "candid", "parti", "polit",
                             "presid", "senat", "congress"],
}


def load_vocab():
    """Load vectorizer vocabulary and shared vocab mask."""
    vec = joblib.load(INPUT_SPEECH_DIR / "05_vectorizer.joblib")
    all_features = vec.get_feature_names_out()
    shared_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    mask = np.load(shared_path) if shared_path.exists() else None
    features = all_features[mask] if mask is not None else all_features
    return features, mask


def load_congress_articles(cong, shared_vocab_mask):
    """Load feature matrix and metadata for one congress.
    Returns X (all articles after sample_idx), meta, is_news boolean array.
    """
    label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
    if not label_path.exists():
        return None, None, None

    meta = pd.read_parquet(label_path, columns=["paper", "year", "is_news"])

    sample_idx_path = INPUT_NEWS_DIR / f"07_sample_idx_cong_{cong}.npy"
    if sample_idx_path.exists():
        idx = np.load(sample_idx_path)
        meta = meta.iloc[idx].reset_index(drop=True)

    features_path = INPUT_NEWS_DIR / f"07_newspaper_features_cong_{cong}.npz"
    X = sp.load_npz(features_path)
    if shared_vocab_mask is not None:
        X = X[:, shared_vocab_mask]

    assert len(meta) == X.shape[0], \
        f"Congress {cong}: meta ({len(meta)}) != X ({X.shape[0]})"

    is_news = meta["is_news"].values
    return X, meta, is_news


def build_training_sample(shared_vocab_mask):
    """Build stratified random subsample of news articles for NMF fitting."""
    print(f"\nBuilding training sample ({SUBSAMPLE_SIZE:,} articles) ...",
          flush=True)
    congresses = cfg.get_congresses()
    rng = np.random.default_rng(RANDOM_STATE)

    # Count news articles per congress
    counts = {}
    for cong in congresses:
        label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        if not label_path.exists():
            continue
        meta = pd.read_parquet(label_path, columns=["is_news"])
        sample_idx_path = INPUT_NEWS_DIR / f"07_sample_idx_cong_{cong}.npy"
        if sample_idx_path.exists():
            idx = np.load(sample_idx_path)
            meta = meta.iloc[idx].reset_index(drop=True)
        counts[cong] = meta["is_news"].sum()
        del meta

    total = sum(counts.values())
    print(f"  Total news articles: {total:,}", flush=True)

    # Proportional sampling
    blocks = []
    for cong in congresses:
        if cong not in counts:
            continue
        n_sample = int(SUBSAMPLE_SIZE * counts[cong] / total)
        X, meta, is_news = load_congress_articles(cong, shared_vocab_mask)
        if X is None:
            continue
        news_idx = np.where(is_news)[0]
        X_news = X[news_idx]
        chosen = rng.choice(X_news.shape[0],
                            size=min(n_sample, X_news.shape[0]),
                            replace=False)
        blocks.append(X_news[chosen])
        print(f"    Congress {cong}: sampled {len(chosen):,} / "
              f"{X_news.shape[0]:,}", flush=True)
        del X, meta, X_news
        gc.collect()

    X_train = sp.vstack(blocks, format="csr")
    del blocks
    gc.collect()
    print(f"  Training sample: {X_train.shape[0]:,} x {X_train.shape[1]:,}",
          flush=True)
    return X_train


def find_topic_by_keywords(model, features, keywords, name):
    """Identify NMF topic best matching given keywords."""
    H = model.components_
    best_topic, best_score = -1, 0
    for k in range(H.shape[0]):
        top_idx = H[k].argsort()[::-1][:20]
        top_words = [features[i] for i in top_idx]
        score = sum(1 for w in top_words
                    for kw in keywords if w.startswith(kw))
        if score > best_score:
            best_score = score
            best_topic = k
    top_idx = H[best_topic].argsort()[::-1][:10]
    top_words = [features[i] for i in top_idx]
    print(f"\n  {name} topic: #{best_topic + 1} (score={best_score})",
          flush=True)
    print(f"  Top words: {', '.join(top_words)}", flush=True)
    return best_topic


def label_topics(model, features):
    """Assign descriptive labels to all topics."""
    H = model.components_
    labels = {}
    for k in range(H.shape[0]):
        top_idx = H[k].argsort()[::-1][:20]
        top_words = [features[i] for i in top_idx]
        best_cat, best_score = f"Topic {k+1}", 0
        for cat, kws in TOPIC_CATEGORIES.items():
            score = sum(1 for w in top_words
                        for kw in kws if w.startswith(kw))
            if score > best_score:
                best_score = score
                best_cat = cat
        labels[k] = {"label": best_cat, "top_words": top_words}
    return labels


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    t0 = time.time()
    congresses = cfg.get_congresses()

    print("=" * 72)
    print("STEP 07b: NMF Content Filter")
    print(f"  Run: {cfg.RUN_NAME}")
    print(f"  Topics: {N_TOPICS}, Subsample: {SUBSAMPLE_SIZE:,}")
    print("=" * 72)

    # 1. Load vocabulary
    features, shared_vocab_mask = load_vocab()
    print(f"\nVocabulary: {len(features):,} features")

    # 2. Build training sample and fit NMF
    X_train = build_training_sample(shared_vocab_mask)

    print(f"\nFitting NMF ({N_TOPICS} topics, max_iter={MAX_ITER}) ...",
          flush=True)
    nmf = NMF(n_components=N_TOPICS, init="nndsvd",
              max_iter=MAX_ITER, random_state=RANDOM_STATE)
    nmf.fit(X_train)
    del X_train
    gc.collect()

    # 3. Identify target topics
    obit_topic = find_topic_by_keywords(nmf, features, OBITUARY_KWS,
                                         "Obituary")
    spanish_topic = find_topic_by_keywords(nmf, features, SPANISH_KWS,
                                            "Spanish")
    exclude_topics = {obit_topic, spanish_topic}

    # 4. Label all topics
    topic_labels = label_topics(nmf, features)
    print("\n  All topic labels:")
    for k, info in topic_labels.items():
        marker = " [EXCLUDE]" if k in exclude_topics else ""
        print(f"    #{k+1}: {info['label']}{marker}  "
              f"({', '.join(info['top_words'][:5])})")

    # 5. Save NMF model and metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "07b_nmf_model.joblib"
    joblib.dump(nmf, model_path, compress=3)
    print(f"\n  NMF model saved -> {model_path}")

    meta_info = {
        "n_topics": N_TOPICS,
        "subsample_size": SUBSAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "obit_topic": int(obit_topic),
        "spanish_topic": int(spanish_topic),
        "topic_labels": {str(k): v for k, v in topic_labels.items()},
    }
    meta_path = MODEL_DIR / "07b_nmf_topic_labels.json"
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)
    print(f"  Topic metadata saved -> {meta_path}")

    # 6. Process each congress: assign topics and save exclusion masks
    print("\nProcessing congresses ...", flush=True)
    total_articles = 0
    total_news = 0
    total_excluded = 0
    total_obit = 0
    total_spanish = 0

    for cong in congresses:
        X, meta, is_news = load_congress_articles(cong, shared_vocab_mask)
        if X is None:
            print(f"  Congress {cong}: SKIPPED (files not found)", flush=True)
            continue

        n_all = len(is_news)
        news_idx = np.where(is_news)[0]
        n_news = len(news_idx)

        # Initialize exclusion mask: False = keep, True = exclude
        nmf_exclude = np.zeros(n_all, dtype=bool)

        if n_news > 0:
            X_news = X[news_idx]
            W = nmf.transform(X_news)
            dominant = W.argmax(axis=1)
            del W

            # Mark obituary and Spanish articles for exclusion
            exclude_mask = np.isin(dominant, list(exclude_topics))
            nmf_exclude[news_idx[exclude_mask]] = True

            n_obit = (dominant == obit_topic).sum()
            n_spanish = (dominant == spanish_topic).sum()
            n_excl = exclude_mask.sum()

            del X_news, dominant
        else:
            n_obit = n_spanish = n_excl = 0

        # Save
        out_path = OUT_DIR / f"07b_nmf_exclude_cong_{cong}.npy"
        np.save(out_path, nmf_exclude)

        total_articles += n_all
        total_news += n_news
        total_excluded += n_excl
        total_obit += n_obit
        total_spanish += n_spanish

        print(f"  Congress {cong}: {n_all:>10,} articles  "
              f"{n_news:>10,} news  "
              f"excluded {n_excl:>8,} "
              f"(obit {n_obit:,}, spanish {n_spanish:,})  "
              f"[{n_excl/n_news*100:.1f}%]" if n_news > 0 else "",
              flush=True)

        del X, meta, is_news, nmf_exclude
        gc.collect()

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: NMF Content Filter")
    print("=" * 72)
    print(f"\n  Total articles:       {total_articles:>12,}")
    print(f"  News articles:        {total_news:>12,}")
    print(f"  NMF excluded:         {total_excluded:>12,}  "
          f"({total_excluded/total_news*100:.1f}% of news)")
    print(f"    - Obituary/funeral: {total_obit:>12,}  "
          f"({total_obit/total_news*100:.1f}%)")
    print(f"    - Spanish-language: {total_spanish:>12,}  "
          f"({total_spanish/total_news*100:.1f}%)")
    print(f"  News after filter:    {total_news - total_excluded:>12,}")
    print(f"\n  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Outputs: {OUT_DIR}/07b_nmf_exclude_cong_*.npy")
    print("=" * 72)
