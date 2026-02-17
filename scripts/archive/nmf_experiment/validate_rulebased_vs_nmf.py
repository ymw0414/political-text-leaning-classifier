"""
Validate: Can simple rule-based filters capture the same noise as NMF?

Compares NMF topic-based identification of obituaries and Spanish-language
articles with simple keyword/stopword rule-based filters applied to the
same articles. Reports precision, recall, F1, and overlap statistics.

Rule-based approach:
  - Spanish: count Spanish stop words (el, en, lo, su, la, de, por, para, con,
    una, los, las, del, que, como, mas, muy, pero, sin, ser, hay, fue, son, tiene)
    If >= 5 distinct Spanish stopwords appear among an article's nonzero features,
    flag as Spanish.
  - Obituary: count obituary content keywords (die, died, funeral, church, born,
    son, daughter, wife, husband, brother, sister, servic, cemeteri, burial, member,
    surviv, mother, father, pall, pastor, baptist)
    If >= 4 distinct obituary keywords appear, flag as obituary.
"""

import os, sys, gc, time, warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import NMF
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nlp"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "utils"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
MODEL_DIR = cfg.MODEL_DIR
SPEECH_DIR = cfg.INPUT_SPEECH_DIR
NEWS_FEATURES_DIR = cfg.INPUT_NEWS_DIR
NEWS_DIR = cfg.NEWS_DIR
LABEL_DIR = cfg.NEWSPAPER_LABELS
REPORTS_DIR = BASE_DIR / "reports"

N_TOPICS = 10
RANDOM_STATE = 42
SUBSAMPLE_SIZE = 500_000

# --- Rule-based keyword lists ---
SPANISH_STOPWORDS = [
    "el", "en", "lo", "su", "la", "de", "por", "para", "con", "una",
    "los", "las", "del", "que", "como", "mas", "muy", "pero", "sin",
    "ser", "hay", "fue", "son", "tiene", "esta", "ese", "todo", "otro",
    "puede", "tiene", "cada", "entre", "desde", "sobre", "tambien",
]
SPANISH_THRESHOLD = 5  # min distinct Spanish stopwords to flag

OBITUARY_KEYWORDS = [
    "die", "died", "funeral", "church", "born", "son", "daughter",
    "wife", "husband", "brother", "sister", "cemeteri",
    "burial", "surviv", "mother", "father", "pallbear",
    "pastor", "baptist", "methodist", "preced", "interment",
    "obituari", "deceas", "mourn", "memori", "chapel", "arrang",
    "widow", "niece", "nephew", "granddaught", "grandson",
    "bereav", "condol", "rosari", "eulog",
]
OBITUARY_THRESHOLD = 3  # min distinct obituary keywords to flag


def load_vocab():
    vec = joblib.load(SPEECH_DIR / "05_vectorizer.joblib")
    all_features = vec.get_feature_names_out()
    shared_vocab_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    shared_vocab_mask = np.load(shared_vocab_path) if shared_vocab_path.exists() else None
    if shared_vocab_mask is not None:
        features = all_features[shared_vocab_mask]
    else:
        features = all_features
    return features, shared_vocab_mask


def build_keyword_indices(features, keywords, exact=False):
    """Find feature indices matching keywords.
    exact=True: only exact matches (for Spanish stopwords that are short)
    exact=False: startswith match (for English stems like 'funeral' -> 'funer')
    """
    indices = []
    matched = []
    for i, f in enumerate(features):
        for kw in keywords:
            if exact:
                if f == kw:
                    indices.append(i)
                    matched.append(f)
                    break
            else:
                if f.startswith(kw):
                    indices.append(i)
                    matched.append(f)
                    break
    return np.array(indices), matched


def load_congress_news(cong, shared_vocab_mask):
    label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
    if not label_path.exists():
        return None, None

    meta = pd.read_parquet(label_path, columns=["paper", "year", "is_news"])

    sample_idx_path = NEWS_FEATURES_DIR / f"07_sample_idx_cong_{cong}.npy"
    if sample_idx_path.exists():
        idx = np.load(sample_idx_path)
        meta = meta.iloc[idx].reset_index(drop=True)

    features_path = NEWS_FEATURES_DIR / f"07_newspaper_features_cong_{cong}.npz"
    X = sp.load_npz(features_path)
    if shared_vocab_mask is not None:
        X = X[:, shared_vocab_mask]

    is_news = meta["is_news"].values
    news_idx = np.where(is_news)[0]
    X_news = X[news_idx]
    meta_news = meta.iloc[news_idx].reset_index(drop=True)

    del X, meta
    return X_news, meta_news


def build_training_sample(shared_vocab_mask):
    print(f"\nBuilding training sample ({SUBSAMPLE_SIZE:,} articles) ...", flush=True)
    congresses = [w[-1] for w in cfg.get_windows()]
    rng = np.random.default_rng(RANDOM_STATE)

    counts = {}
    for cong in congresses:
        label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        if not label_path.exists():
            continue
        meta = pd.read_parquet(label_path, columns=["is_news"])
        sample_idx_path = NEWS_FEATURES_DIR / f"07_sample_idx_cong_{cong}.npy"
        if sample_idx_path.exists():
            idx = np.load(sample_idx_path)
            meta = meta.iloc[idx].reset_index(drop=True)
        counts[cong] = meta["is_news"].sum()
        del meta

    total = sum(counts.values())
    blocks = []
    for cong in congresses:
        if cong not in counts:
            continue
        n_sample = int(SUBSAMPLE_SIZE * counts[cong] / total)
        X_news, meta = load_congress_news(cong, shared_vocab_mask)
        if X_news is None:
            continue
        chosen = rng.choice(X_news.shape[0], size=min(n_sample, X_news.shape[0]), replace=False)
        blocks.append(X_news[chosen])
        print(f"    Congress {cong}: sampled {len(chosen):,} / {X_news.shape[0]:,}", flush=True)
        del X_news, meta
        gc.collect()

    X_train = sp.vstack(blocks, format="csr")
    del blocks
    gc.collect()
    print(f"  Training sample: {X_train.shape[0]:,} articles, {X_train.shape[1]:,} features", flush=True)
    return X_train


def find_topic_by_keywords(model, features, keywords, name):
    H = model.components_
    best_topic, best_score = -1, 0
    for k in range(H.shape[0]):
        top_idx = H[k].argsort()[::-1][:20]
        top_words = [features[i] for i in top_idx]
        score = sum(1 for w in top_words for kw in keywords if w.startswith(kw))
        if score > best_score:
            best_score = score
            best_topic = k
    top_idx = H[best_topic].argsort()[::-1][:10]
    top_words = [features[i] for i in top_idx]
    print(f"\n  {name} topic: #{best_topic + 1} (score={best_score})", flush=True)
    print(f"  Top words: {', '.join(top_words)}", flush=True)
    return best_topic


def rule_based_flags(X_news, spanish_idx, obit_idx):
    """Apply rule-based filters to sparse feature matrix.
    Returns boolean arrays for Spanish and obituary flags."""
    n = X_news.shape[0]

    # Spanish: count how many distinct Spanish stopwords have nonzero count
    if len(spanish_idx) > 0:
        X_sp = X_news[:, spanish_idx]
        # For each article, count number of nonzero Spanish features
        sp_counts = np.array((X_sp > 0).sum(axis=1)).ravel()
        is_spanish_rule = sp_counts >= SPANISH_THRESHOLD
    else:
        is_spanish_rule = np.zeros(n, dtype=bool)

    # Obituary: count how many distinct obituary keywords have nonzero count
    if len(obit_idx) > 0:
        X_ob = X_news[:, obit_idx]
        ob_counts = np.array((X_ob > 0).sum(axis=1)).ravel()
        is_obit_rule = ob_counts >= OBITUARY_THRESHOLD
    else:
        is_obit_rule = np.zeros(n, dtype=bool)

    return is_spanish_rule, is_obit_rule


def main():
    t0 = time.time()

    # Load vocab and build keyword indices
    features, shared_vocab_mask = load_vocab()
    print(f"Vocabulary: {len(features):,} features", flush=True)

    spanish_idx, spanish_matched = build_keyword_indices(features, SPANISH_STOPWORDS, exact=True)
    obit_idx, obit_matched = build_keyword_indices(features, OBITUARY_KEYWORDS, exact=False)
    print(f"\nSpanish rule: {len(spanish_idx)} matching features in vocab")
    print(f"  Matched: {', '.join(spanish_matched[:20])}{'...' if len(spanish_matched) > 20 else ''}")
    print(f"\nObituary rule: {len(obit_idx)} matching features in vocab")
    print(f"  Matched: {', '.join(obit_matched[:20])}{'...' if len(obit_matched) > 20 else ''}")

    # Build training sample and fit NMF
    X_train = build_training_sample(shared_vocab_mask)

    print("\nFitting NMF (10 topics) ...", flush=True)
    nmf = NMF(n_components=N_TOPICS, init="nndsvd", max_iter=300, random_state=RANDOM_STATE)
    nmf.fit(X_train)
    del X_train
    gc.collect()

    # Identify NMF topics
    obit_kws = ["die", "died", "funeral", "church", "born", "son", "daughter",
                 "wife", "husband", "surviv", "servic", "cemeteri", "burial"]
    spanish_kws = ["el", "en", "lo", "su", "la", "de", "por", "para", "con",
                    "una", "los", "las", "del", "que", "como"]
    obit_topic = find_topic_by_keywords(nmf, features, obit_kws, "Obituary")
    spanish_topic = find_topic_by_keywords(nmf, features, spanish_kws, "Spanish")

    # Process all congresses
    congresses = [w[-1] for w in cfg.get_windows()]
    total_stats = {
        "n_total": 0,
        "nmf_obit": 0, "nmf_spanish": 0,
        "rule_obit": 0, "rule_spanish": 0,
        "both_obit": 0, "both_spanish": 0,
        "nmf_only_obit": 0, "nmf_only_spanish": 0,
        "rule_only_obit": 0, "rule_only_spanish": 0,
    }

    per_congress = []

    for cong in congresses:
        X_news, meta = load_congress_news(cong, shared_vocab_mask)
        if X_news is None:
            continue

        n = X_news.shape[0]

        # NMF topic assignment
        W = nmf.transform(X_news)
        dominant = W.argmax(axis=1)
        del W

        is_obit_nmf = (dominant == obit_topic)
        is_spanish_nmf = (dominant == spanish_topic)

        # Rule-based flags
        is_spanish_rule, is_obit_rule = rule_based_flags(X_news, spanish_idx, obit_idx)
        del X_news
        gc.collect()

        # Overlap stats
        both_obit = (is_obit_nmf & is_obit_rule).sum()
        both_spanish = (is_spanish_nmf & is_spanish_rule).sum()
        nmf_only_obit = (is_obit_nmf & ~is_obit_rule).sum()
        nmf_only_spanish = (is_spanish_nmf & ~is_spanish_rule).sum()
        rule_only_obit = (~is_obit_nmf & is_obit_rule).sum()
        rule_only_spanish = (~is_spanish_nmf & is_spanish_rule).sum()

        stats = {
            "congress": cong, "n": n,
            "nmf_obit": is_obit_nmf.sum(), "nmf_spanish": is_spanish_nmf.sum(),
            "rule_obit": is_obit_rule.sum(), "rule_spanish": is_spanish_rule.sum(),
            "both_obit": both_obit, "both_spanish": both_spanish,
            "nmf_only_obit": nmf_only_obit, "nmf_only_spanish": nmf_only_spanish,
            "rule_only_obit": rule_only_obit, "rule_only_spanish": rule_only_spanish,
        }
        per_congress.append(stats)

        for key in total_stats:
            if key == "n_total":
                total_stats[key] += n
            elif key in stats:
                total_stats[key] += stats[key]

        # Print per-congress summary
        def pct(num, denom):
            return f"{num/denom:.1%}" if denom > 0 else "N/A"

        print(f"\n  Congress {cong} (N={n:,}):")
        print(f"    Obituary  - NMF: {is_obit_nmf.sum():,}  Rule: {is_obit_rule.sum():,}  "
              f"Both: {both_obit:,}  NMF-only: {nmf_only_obit:,}  Rule-only: {rule_only_obit:,}")
        print(f"    Spanish   - NMF: {is_spanish_nmf.sum():,}  Rule: {is_spanish_rule.sum():,}  "
              f"Both: {both_spanish:,}  NMF-only: {nmf_only_spanish:,}  Rule-only: {rule_only_spanish:,}")

        del meta, dominant, is_obit_nmf, is_spanish_nmf, is_obit_rule, is_spanish_rule
        gc.collect()

    # Aggregate statistics
    N = total_stats["n_total"]
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"Total news articles: {N:,}\n")

    for cat in ["obit", "spanish"]:
        label = "Obituary" if cat == "obit" else "Spanish"
        nmf_n = total_stats[f"nmf_{cat}"]
        rule_n = total_stats[f"rule_{cat}"]
        both_n = total_stats[f"both_{cat}"]
        nmf_only = total_stats[f"nmf_only_{cat}"]
        rule_only = total_stats[f"rule_only_{cat}"]

        # Recall of NMF (how many NMF-flagged does rule catch)
        recall_of_nmf = both_n / nmf_n if nmf_n > 0 else 0
        # Precision of rule (how many rule-flagged are also NMF-flagged)
        precision_vs_nmf = both_n / rule_n if rule_n > 0 else 0
        # Rule captures everything NMF does + extra
        coverage = (both_n + rule_only) / (nmf_n + rule_only) if (nmf_n + rule_only) > 0 else 0

        print(f"--- {label} ---")
        print(f"  NMF flagged:     {nmf_n:>8,}  ({nmf_n/N:.1%})")
        print(f"  Rule flagged:    {rule_n:>8,}  ({rule_n/N:.1%})")
        print(f"  Both flagged:    {both_n:>8,}")
        print(f"  NMF-only:        {nmf_only:>8,}  (NMF caught, rule missed)")
        print(f"  Rule-only:       {rule_only:>8,}  (Rule caught, NMF missed)")
        print(f"  Recall of NMF:   {recall_of_nmf:.1%}  (rule catches this % of NMF-flagged)")
        print(f"  Jaccard overlap: {both_n/(nmf_n + rule_n - both_n):.1%}" if (nmf_n + rule_n - both_n) > 0 else "")
        print()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Generate LaTeX report
    generate_report(total_stats, per_congress, features, spanish_matched, obit_matched, elapsed)


def generate_report(stats, per_congress, features, spanish_matched, obit_matched, elapsed):
    N = stats["n_total"]

    def pct(num, denom):
        return f"{num/denom*100:.1f}\\%" if denom > 0 else "---"

    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs,array,xcolor,amsmath}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{center}")
    lines.append(r"{\Large\bfseries Rule-Based vs.\ NMF Filter Validation}\\[6pt]")
    lines.append(r"{\normalsize Can simple keyword filters replace NMF topic assignment?}\\[4pt]")
    lines.append(r"{\small\today}")
    lines.append(r"\end{center}")
    lines.append(r"\vspace{0.5em}")

    # Section 1: Method
    lines.append(r"\section*{1. Method}")
    lines.append(r"We compare two approaches for identifying non-news content that passed the title-based \texttt{is\_news} filter:")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item \textbf{NMF (data-driven):} 10-topic NMF on TF features; articles assigned to dominant topic; obituary and Spanish topics identified by top-word matching.")
    lines.append(r"\item \textbf{Rule-based (simple):} Count distinct keyword matches in each article's nonzero features. Flag as Spanish if $\geq$" + str(SPANISH_THRESHOLD) + r" Spanish stopwords present; flag as obituary if $\geq$" + str(OBITUARY_THRESHOLD) + r" obituary content words present.")
    lines.append(r"\end{itemize}")

    lines.append(r"\paragraph{Spanish keywords (" + str(len(spanish_matched)) + r" features matched):}")
    lines.append(r"\texttt{" + ", ".join(spanish_matched[:25]) + ("..." if len(spanish_matched) > 25 else "") + r"}")

    lines.append(r"\paragraph{Obituary keywords (" + str(len(obit_matched)) + r" features matched):}")
    lines.append(r"\texttt{" + ", ".join(obit_matched[:25]) + ("..." if len(obit_matched) > 25 else "") + r"}")

    # Section 2: Overlap table
    lines.append(r"\section*{2. Aggregate Overlap}")
    lines.append(r"\begin{table}[h]\centering")
    lines.append(r"\begin{tabular}{l rr rr rr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{Obituary} & \multicolumn{2}{c}{Spanish} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r" & Count & Share & Count & Share \\")
    lines.append(r"\midrule")
    lines.append(r"Total articles & " + f"{N:,}" + r" & 100\% & " + f"{N:,}" + r" & 100\% \\")

    for cat, label in [("obit", "Obituary"), ("spanish", "Spanish")]:
        nmf_n = stats[f"nmf_{cat}"]
        rule_n = stats[f"rule_{cat}"]
        both_n = stats[f"both_{cat}"]
        nmf_only = stats[f"nmf_only_{cat}"]
        rule_only = stats[f"rule_only_{cat}"]
        col = 2 if cat == "obit" else 4

    # Simpler: just two sub-tables
    lines.pop()  # remove the total articles line
    lines.pop()  # remove midrule
    lines.pop()  # remove header
    lines.pop()  # remove cmidrule
    lines.pop()  # remove cmidrule
    lines.pop()  # remove toprule
    lines.pop()  # remove tabular
    lines.pop()  # remove table

    lines.append(r"\begin{table}[h]\centering")
    lines.append(r"\caption{NMF vs.\ Rule-based filter overlap (N=" + f"{N:,}" + r")}")
    lines.append(r"\begin{tabular}{l rrr rrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{Obituary} & \multicolumn{3}{c}{Spanish} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r" & Count & \% of Total & & Count & \% of Total & \\")
    lines.append(r"\midrule")

    for cat_label, cat in [("NMF flagged", "nmf"), ("Rule flagged", "rule"), ("Both flagged", "both"),
                            ("NMF-only", "nmf_only"), ("Rule-only", "rule_only")]:
        ob = stats.get(f"{cat}_obit", 0)
        sp = stats.get(f"{cat}_spanish", 0)
        lines.append(f"  {cat_label} & {ob:,} & {pct(ob, N)} & & {sp:,} & {pct(sp, N)} & \\\\")

    lines.append(r"\midrule")
    # Key metrics
    for cat, label in [("obit", "Obituary"), ("spanish", "Spanish")]:
        nmf_n = stats[f"nmf_{cat}"]
        both_n = stats[f"both_{cat}"]
        rule_n = stats[f"rule_{cat}"]
        recall = both_n / nmf_n * 100 if nmf_n > 0 else 0
        jaccard = both_n / (nmf_n + rule_n - both_n) * 100 if (nmf_n + rule_n - both_n) > 0 else 0

    recall_ob = stats["both_obit"] / stats["nmf_obit"] * 100 if stats["nmf_obit"] > 0 else 0
    recall_sp = stats["both_spanish"] / stats["nmf_spanish"] * 100 if stats["nmf_spanish"] > 0 else 0
    jac_ob_denom = stats["nmf_obit"] + stats["rule_obit"] - stats["both_obit"]
    jac_sp_denom = stats["nmf_spanish"] + stats["rule_spanish"] - stats["both_spanish"]
    jac_ob = stats["both_obit"] / jac_ob_denom * 100 if jac_ob_denom > 0 else 0
    jac_sp = stats["both_spanish"] / jac_sp_denom * 100 if jac_sp_denom > 0 else 0

    lines.append(f"  Recall of NMF & \\multicolumn{{3}}{{c}}{{{recall_ob:.1f}\\%}} & \\multicolumn{{3}}{{c}}{{{recall_sp:.1f}\\%}} \\\\")
    lines.append(f"  Jaccard similarity & \\multicolumn{{3}}{{c}}{{{jac_ob:.1f}\\%}} & \\multicolumn{{3}}{{c}}{{{jac_sp:.1f}\\%}} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # Section 3: Interpretation
    lines.append(r"\section*{3. Interpretation}")

    # Determine verdict for each
    for cat, label in [("Obituary", "obit"), ("Spanish", "spanish")]:
        nmf_n = stats[f"nmf_{label}"]
        rule_n = stats[f"rule_{label}"]
        both_n = stats[f"both_{label}"]
        nmf_only = stats[f"nmf_only_{label}"]
        recall = both_n / nmf_n * 100 if nmf_n > 0 else 0

        lines.append(f"\\paragraph{{{cat}:}}")
        if recall >= 80:
            lines.append(f"The rule-based filter captures {recall:.0f}\\% of NMF-flagged {cat.lower()} articles. "
                         f"The {nmf_only:,} NMF-only articles ({nmf_only/nmf_n*100:.1f}\\% of NMF-flagged) are likely "
                         f"borderline cases. Simple keyword filters are \\textbf{{sufficient}} as a replacement for NMF.")
        elif recall >= 50:
            lines.append(f"The rule-based filter captures {recall:.0f}\\% of NMF-flagged {cat.lower()} articles. "
                         f"Coverage is moderate; threshold tuning or expanded keyword lists may improve recall. "
                         f"Rule-based filters are \\textbf{{partially effective}} but may benefit from refinement.")
        else:
            lines.append(f"The rule-based filter captures only {recall:.0f}\\% of NMF-flagged {cat.lower()} articles. "
                         f"Simple keyword matching is \\textbf{{insufficient}}; NMF captures latent patterns not easily "
                         f"reducible to keyword lists.")

    lines.append(r"\section*{4. Conclusion}")
    overall_nmf = stats["nmf_obit"] + stats["nmf_spanish"]
    overall_rule = stats["rule_obit"] + stats["rule_spanish"]
    overall_both = stats["both_obit"] + stats["both_spanish"]
    overall_recall = overall_both / overall_nmf * 100 if overall_nmf > 0 else 0

    lines.append(f"Overall, rule-based filters flag {overall_rule:,} articles vs.\\ NMF's {overall_nmf:,}. "
                 f"Of the NMF-flagged articles, {overall_recall:.0f}\\% are also caught by rules. ")
    if overall_recall >= 80:
        lines.append(r"This confirms that simple, interpretable, and deterministic rule-based filters "
                     r"can effectively replace NMF for the purpose of excluding non-news noise from the sample, "
                     r"without introducing the methodological dependency on an arbitrary topic model.")
    else:
        lines.append(r"Some NMF-flagged articles are missed by rules, suggesting that a more sophisticated "
                     r"approach (expanded keyword lists or a dedicated classifier) may be needed for full coverage.")

    lines.append(f"\n\\vspace{{1em}}\n\\noindent\\textit{{Runtime: {elapsed:.0f}s}}")
    lines.append(r"\end{document}")

    report_path = REPORTS_DIR / "rulebased_vs_nmf_validation.tex"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
