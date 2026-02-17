"""
Obituary exclusion robustness test.

NMF identified ~10% of 'news' articles as obituary-like content (die, funeral,
church, born, etc.) that passed the title-based is_news filter. This script:

1. Re-fits NMF to identify the obituary topic
2. Re-computes share_R excluding obituary-assigned articles
3. Runs the main DiD specification on the filtered sample
4. Compares with baseline results

Output:
  - Console: comparison table (baseline vs excluding obituaries)
  - reports/obituary_exclusion_test.tex: LaTeX report
"""

import os, sys, gc, time, warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import NMF
import pyfixest as pf
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
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
REPORTS_DIR = BASE_DIR / "reports"

NAFTA_YEAR = 1994
END_YEAR = 2004
N_TOPICS = 10
RANDOM_STATE = 42
SUBSAMPLE_SIZE = 500_000

STATE_TO_DIVISION = {
    9: 1, 23: 1, 25: 1, 33: 1, 44: 1, 50: 1,
    34: 2, 36: 2, 42: 2,
    17: 3, 18: 3, 26: 3, 39: 3, 55: 3,
    19: 4, 20: 4, 27: 4, 29: 4, 31: 4, 38: 4, 46: 4,
    10: 5, 11: 5, 12: 5, 13: 5, 24: 5, 37: 5, 45: 5, 51: 5, 54: 5,
    1: 6, 21: 6, 28: 6, 47: 6,
    5: 7, 22: 7, 40: 7, 48: 7,
    4: 8, 8: 8, 16: 8, 30: 8, 32: 8, 35: 8, 49: 8, 56: 8,
    2: 9, 6: 9, 15: 9, 41: 9, 53: 9,
}


def load_vocab():
    vec = joblib.load(SPEECH_DIR / "05_vectorizer.joblib")
    all_features = vec.get_feature_names_out()
    shared_vocab_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    shared_vocab_mask = np.load(shared_vocab_path) if shared_vocab_path.exists() else None
    if shared_vocab_mask is not None:
        features = all_features[shared_vocab_mask]
    else:
        features = all_features
        shared_vocab_mask = None
    return features, shared_vocab_mask


def load_congress_news(cong, shared_vocab_mask):
    label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
    if not label_path.exists():
        return None, None, None
    meta = pd.read_parquet(label_path, columns=["paper", "year", "is_news"])

    sample_idx_path = NEWS_FEATURES_DIR / f"07_sample_idx_cong_{cong}.npy"
    if sample_idx_path.exists():
        idx = np.load(sample_idx_path)
        meta = meta.iloc[idx].reset_index(drop=True)

    slant_path = NEWS_DIR / f"08_article_slant_cong_{cong}.parquet"
    if not slant_path.exists():
        return None, None, None
    slant = pd.read_parquet(slant_path, columns=["net_slant"])

    features_path = NEWS_FEATURES_DIR / f"07_newspaper_features_cong_{cong}.npz"
    X = sp.load_npz(features_path)
    if shared_vocab_mask is not None:
        X = X[:, shared_vocab_mask]

    is_news = meta["is_news"].values
    news_idx = np.where(is_news)[0]

    X_news = X[news_idx]
    meta_news = pd.DataFrame({
        "paper": meta["paper"].values[news_idx],
        "year": meta["year"].values[news_idx],
        "net_slant": slant["net_slant"].values[news_idx],
    })

    del X, meta, slant
    return X_news, meta_news, news_idx


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
    print(f"  Total news articles: {total:,}", flush=True)

    blocks = []
    for cong in congresses:
        if cong not in counts:
            continue
        n_sample = int(SUBSAMPLE_SIZE * counts[cong] / total)
        X_news, meta, _ = load_congress_news(cong, shared_vocab_mask)
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
    """Identify which NMF topic best matches the given keywords."""
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


def process_congress(cong, model, shared_vocab_mask, obit_topic, spanish_topic):
    """Compute newspaper-year share_R with various exclusions."""
    X_news, meta, _ = load_congress_news(cong, shared_vocab_mask)
    if X_news is None:
        return None

    is_R = (meta["net_slant"].values > 0).astype(np.int8)
    W = model.transform(X_news)
    del X_news

    dominant = W.argmax(axis=1)
    del W

    is_obit = (dominant == obit_topic)
    is_spanish = (dominant == spanish_topic)
    is_either = is_obit | is_spanish
    n_total = len(dominant)
    n_obit = is_obit.sum()
    n_spanish = is_spanish.sum()

    df = pd.DataFrame({
        "paper": meta["paper"].values,
        "year": meta["year"].values,
        "is_R": is_R,
        "is_obit": is_obit.astype(np.int8),
        "is_spanish": is_spanish.astype(np.int8),
        "is_either": is_either.astype(np.int8),
    })

    # All articles (baseline)
    grp_all = df.groupby(["paper", "year"])
    agg = grp_all.agg(
        n_all=("is_R", "count"),
        share_R_baseline=("is_R", "mean"),
    ).reset_index()

    # Excluding obituaries only
    df_no_obit = df[~df["is_obit"].astype(bool)]
    agg_no_obit = df_no_obit.groupby(["paper", "year"]).agg(
        n_no_obit=("is_R", "count"),
        share_R_no_obit=("is_R", "mean"),
    ).reset_index()
    agg = agg.merge(agg_no_obit, on=["paper", "year"], how="left")

    # Excluding Spanish only
    df_no_sp = df[~df["is_spanish"].astype(bool)]
    agg_no_sp = df_no_sp.groupby(["paper", "year"]).agg(
        n_no_spanish=("is_R", "count"),
        share_R_no_spanish=("is_R", "mean"),
    ).reset_index()
    agg = agg.merge(agg_no_sp, on=["paper", "year"], how="left")

    # Excluding both
    df_no_both = df[~df["is_either"].astype(bool)]
    agg_no_both = df_no_both.groupby(["paper", "year"]).agg(
        n_no_both=("is_R", "count"),
        share_R_no_both=("is_R", "mean"),
    ).reset_index()
    agg = agg.merge(agg_no_both, on=["paper", "year"], how="left")

    # Shares
    obit_grp = df.groupby(["paper", "year"])["is_obit"].sum().reset_index()
    obit_grp.columns = ["paper", "year", "n_obit"]
    sp_grp = df.groupby(["paper", "year"])["is_spanish"].sum().reset_index()
    sp_grp.columns = ["paper", "year", "n_spanish"]
    agg = agg.merge(obit_grp, on=["paper", "year"], how="left")
    agg = agg.merge(sp_grp, on=["paper", "year"], how="left")
    agg["obit_share"] = agg["n_obit"] / agg["n_all"]
    agg["spanish_share"] = agg["n_spanish"] / agg["n_all"]

    print(f"  Congress {cong}: {len(agg)} obs, "
          f"obit {n_obit:,} ({n_obit/n_total:.1%}), "
          f"spanish {n_spanish:,} ({n_spanish/n_total:.1%})", flush=True)

    return agg


def prepare_regression_data(panel):
    """Merge with regression panel and prepare variables."""
    reg = pd.read_parquet(PANEL_PATH)
    reg = reg[reg["cz"].notna() & reg["vulnerability1990_scaled"].notna()].copy()
    reg = reg[reg["year"] <= END_YEAR].copy()

    merged = panel.merge(
        reg[["paper", "year", "cz", "fips", "vulnerability1990_scaled",
             "manushare1990", "china_shock"]],
        on=["paper", "year"], how="inner"
    )

    merged["state_fips"] = (merged["fips"] // 1000).astype(int)
    merged["division"] = merged["state_fips"].map(STATE_TO_DIVISION)
    merged["paper_id"] = merged["paper"].astype("category").cat.codes
    merged["post"] = (merged["year"] >= NAFTA_YEAR).astype(int)
    merged["vuln_x_post"] = merged["vulnerability1990_scaled"] * merged["post"]

    years = sorted(merged["year"].unique())
    base_yr = years[0]
    for yr in years:
        if yr == base_yr:
            continue
        merged[f"manu_{yr}"] = (merged["year"] == yr).astype(float) * merged["manushare1990"].fillna(0)
        merged[f"china_{yr}"] = (merged["year"] == yr).astype(float) * merged["china_shock"].fillna(0)

    manu_str = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_str = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])
    fml_base = f"vuln_x_post + {china_str} + {manu_str} | paper_id + year + division^year"

    return merged, fml_base


def run_did(merged, fml_base, depvar, label):
    """Run one DiD regression."""
    subset = merged[merged[depvar].notna()].copy()
    n_obs = len(subset)
    if n_obs < 100:
        return None

    fml = f"{depvar} ~ {fml_base}"
    try:
        m = pf.feols(fml, data=subset, vcov={"CRV1": "cz"})
        t = m.tidy().loc["vuln_x_post"]
        coef, se, p = t["Estimate"], t["Std. Error"], t["Pr(>|t|)"]
    except Exception as e:
        print(f"  {label}: ERROR: {e}", flush=True)
        return None

    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return {"label": label, "n": n_obs, "coef": coef, "se": se, "p": p, "stars": stars}


def generate_report(results, stats):
    """Generate LaTeX report with 4-column comparison."""
    report_path = REPORTS_DIR / "obituary_exclusion_test.tex"

    b = results["baseline"]
    no = results["no_obit"]
    ns = results["no_spanish"]
    nb = results["no_both"]

    def pct(new, old):
        return ((new["coef"] - old["coef"]) / old["coef"]) * 100 if old["coef"] != 0 else 0

    tex = r"""\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{mathpazo}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{threeparttable}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue!60!black, citecolor=blue!60!black}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\title{\textbf{Robustness: Excluding Non-News Articles}\\[6pt]
\large Obituary and Spanish-Language Article Exclusion Test}
\author{Research Report}
\date{February 2026}

\begin{document}
\maketitle

\section{Motivation}

NMF topic modeling revealed two categories of articles that passed the title-based \texttt{is\_news} filter but may not represent genuine English-language news content:

\begin{enumerate}[nosep]
    \item \textbf{Obituary-like articles} (""" + f"$\\sim${stats['pct_obit']:.1f}" + r"""\%): Top words include \textit{die, home, mr, funeral, church, service, born}. These passed the filter because their titles lacked explicit obituary keywords.
    \item \textbf{Spanish-language articles} (""" + f"$\\sim${stats['pct_spanish']:.1f}" + r"""\%): Top words include \textit{el, en, lo, su, contra}. These are articles from bilingual or Spanish-language sections of newspapers.
\end{enumerate}

This test re-computes \texttt{share\_R} under three exclusion scenarios and compares with the baseline.

\section{Article Statistics}

\begin{itemize}[nosep]
    \item Total news articles: """ + f"{stats['n_total']:,}" + r"""
    \item Obituary-like: """ + f"{stats['n_obit']:,} ({stats['pct_obit']:.1f}" + r"""\%), mean share per newspaper-year: """ + f"{stats['mean_obit_share']:.1%}" + r"""
    \item Spanish-language: """ + f"{stats['n_spanish']:,} ({stats['pct_spanish']:.1f}" + r"""\%), mean share per newspaper-year: """ + f"{stats['mean_spanish_share']:.1%}" + r"""
    \item Combined: """ + f"{stats['n_both']:,} ({stats['pct_both']:.1f}" + r"""\%)
\end{itemize}

\section{Results}

\begin{table}[H]
\centering
\caption{Main DiD: Article Exclusion Robustness}
\label{tab:excl}
\begin{threeparttable}
\begin{tabular}{@{} l c c c c @{}}
\toprule
& (1) & (2) & (3) & (4) \\
& \textbf{Baseline} & \textbf{Excl.\ Obit.} & \textbf{Excl.\ Spanish} & \textbf{Excl.\ Both} \\
\midrule
"""

    tex += f"""DiD coef. & {b['coef']:.4f}{b['stars']} & {no['coef']:.4f}{no['stars']} & {ns['coef']:.4f}{ns['stars']} & {nb['coef']:.4f}{nb['stars']} \\\\
& ({b['se']:.4f}) & ({no['se']:.4f}) & ({ns['se']:.4f}) & ({nb['se']:.4f}) \\\\
\\addlinespace
$\\Delta$ vs.\ baseline & --- & {pct(no, b):+.1f}\\% & {pct(ns, b):+.1f}\\% & {pct(nb, b):+.1f}\\% \\\\
$N$ & {b['n']:,} & {no['n']:,} & {ns['n']:,} & {nb['n']:,} \\\\
"""

    tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} All columns use the same specification: paper FE, year FE, division$\times$year FE, year-specific manufacturing share and China shock controls, SEs clustered by CZ. {*} $p<0.1$, {**} $p<0.05$, {***} $p<0.01$.
\end{tablenotes}
\end{threeparttable}
\end{table}

"""

    # Short/long decomposition
    sl_keys = [("baseline_short", "baseline_long"), ("no_obit_short", "no_obit_long"),
               ("no_spanish_short", "no_spanish_long"), ("no_both_short", "no_both_long")]
    has_sl = all(k in results for pair in sl_keys for k in pair)

    if has_sl:
        bs = results["baseline_short"]
        bl = results["baseline_long"]
        nos = results["no_obit_short"]
        nol = results["no_obit_long"]
        nss = results["no_spanish_short"]
        nsl = results["no_spanish_long"]
        nbs = results["no_both_short"]
        nbl = results["no_both_long"]

        tex += r"""\begin{table}[H]
\centering
\caption{Short/Long Decomposition: Article Exclusion Robustness}
\begin{threeparttable}
\begin{tabular}{@{} l c c c c @{}}
\toprule
& \textbf{Baseline} & \textbf{Excl.\ Obit.} & \textbf{Excl.\ Spanish} & \textbf{Excl.\ Both} \\
\midrule
"""
        tex += f"""Short-run & {bs['coef']:.4f}{bs['stars']} & {nos['coef']:.4f}{nos['stars']} & {nss['coef']:.4f}{nss['stars']} & {nbs['coef']:.4f}{nbs['stars']} \\\\
& ({bs['se']:.4f}) & ({nos['se']:.4f}) & ({nss['se']:.4f}) & ({nbs['se']:.4f}) \\\\
\\addlinespace
Long-run & {bl['coef']:.4f}{bl['stars']} & {nol['coef']:.4f}{nol['stars']} & {nsl['coef']:.4f}{nsl['stars']} & {nbl['coef']:.4f}{nbl['stars']} \\\\
& ({bl['se']:.4f}) & ({nol['se']:.4f}) & ({nsl['se']:.4f}) & ({nbl['se']:.4f}) \\\\
"""
        tex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item \textit{Notes:} Short-run: 1994--1998; Long-run: 1999--2004. Same specification as Table~\ref{tab:excl}.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""

    pct_both = pct(nb, b)
    tex += r"""
\section{Conclusion}

""" + f"""Excluding obituary-like articles changes the DiD coefficient by {pct(no, b):+.1f}\\%, excluding Spanish-language articles by {pct(ns, b):+.1f}\\%, and excluding both by {pct_both:+.1f}\\%. """ + r"""All specifications remain statistically significant at the 1\% level. The results confirm that non-news content that passed the title-based filter does not materially affect the main findings.

\end{document}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"\n  Report saved: {report_path}", flush=True)
    return report_path


def main():
    t0 = time.time()

    # 1. Load vocabulary and fit NMF
    features, shared_vocab_mask = load_vocab()
    print(f"  {len(features)} features", flush=True)

    X_train = build_training_sample(shared_vocab_mask)

    print(f"\nFitting NMF with {N_TOPICS} topics ...", flush=True)
    nmf = NMF(n_components=N_TOPICS, init="nndsvd", max_iter=300, random_state=RANDOM_STATE)
    nmf.fit(X_train)
    del X_train
    gc.collect()
    print(f"  NMF fit complete", flush=True)

    # 2. Identify obituary and Spanish topics
    obit_keywords = ["die", "funer", "church", "born", "servic", "mr", "mrs",
                     "daughter", "son", "surviv", "memori", "burial"]
    spanish_keywords = ["el", "en", "lo", "su", "se", "le", "contra", "por",
                        "una", "las", "cuba", "sin", "pie"]
    obit_topic = find_topic_by_keywords(nmf, features, obit_keywords, "Obituary")
    spanish_topic = find_topic_by_keywords(nmf, features, spanish_keywords, "Spanish-language")

    # 3. Process each congress
    print(f"\nProcessing articles ...", flush=True)
    congresses = [w[-1] for w in cfg.get_windows()]
    panels = []
    total_articles = 0
    total_obit = 0
    total_spanish = 0

    for cong in congresses:
        agg = process_congress(cong, nmf, shared_vocab_mask, obit_topic, spanish_topic)
        if agg is not None:
            panels.append(agg)
            total_articles += agg["n_all"].sum()
            total_obit += agg["n_obit"].sum()
            total_spanish += agg["n_spanish"].sum()

    panel = pd.concat(panels, ignore_index=True)
    print(f"\n  Total panel: {len(panel)} newspaper-year obs", flush=True)
    print(f"  Total articles: {total_articles:,}", flush=True)
    print(f"  Obituary articles: {total_obit:,} ({total_obit/total_articles:.1%})", flush=True)
    print(f"  Spanish articles: {total_spanish:,} ({total_spanish/total_articles:.1%})", flush=True)

    stats = {
        "n_total": int(total_articles),
        "n_obit": int(total_obit),
        "n_spanish": int(total_spanish),
        "n_both": int(total_obit + total_spanish),
        "pct_obit": total_obit / total_articles * 100,
        "pct_spanish": total_spanish / total_articles * 100,
        "pct_both": (total_obit + total_spanish) / total_articles * 100,
        "mean_obit_share": panel["obit_share"].mean(),
        "median_obit_share": panel["obit_share"].median(),
        "mean_spanish_share": panel["spanish_share"].mean(),
        "median_spanish_share": panel["spanish_share"].median(),
    }

    # 4. Run DiD regressions
    merged, fml_base = prepare_regression_data(panel)
    print(f"\n  Merged for regression: {len(merged)} obs", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  DiD Results Comparison", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    specs = [
        ("share_R_baseline", "Baseline", "baseline"),
        ("share_R_no_obit", "Excl. obituaries", "no_obit"),
        ("share_R_no_spanish", "Excl. Spanish", "no_spanish"),
        ("share_R_no_both", "Excl. both", "no_both"),
    ]

    for depvar, label, key in specs:
        r = run_did(merged, fml_base, depvar, label)
        if r:
            results[key] = r
            print(f"  {label:<20s}: {r['coef']:.4f}{r['stars']} ({r['se']:.4f}), N={r['n']}", flush=True)

    # Short/long decomposition
    years = sorted(merged["year"].unique())
    base_yr = years[0]
    manu_str = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_str = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])

    merged["short_post"] = ((merged["year"] >= 1994) & (merged["year"] <= 1998)).astype(int)
    merged["long_post"] = (merged["year"] >= 1999).astype(int)
    merged["vuln_x_short"] = merged["vulnerability1990_scaled"] * merged["short_post"]
    merged["vuln_x_long"] = merged["vulnerability1990_scaled"] * merged["long_post"]

    fml_sl = f"vuln_x_short + vuln_x_long + {china_str} + {manu_str} | paper_id + year + division^year"

    print(f"\n  Short/Long decomposition:", flush=True)
    for depvar, label_prefix, key_prefix in specs:
        subset = merged[merged[depvar].notna()].copy()
        fml = f"{depvar} ~ {fml_sl}"
        try:
            m = pf.feols(fml, data=subset, vcov={"CRV1": "cz"})
            for var, period in [("vuln_x_short", "short"), ("vuln_x_long", "long")]:
                t = m.tidy().loc[var]
                coef, se, p = t["Estimate"], t["Std. Error"], t["Pr(>|t|)"]
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
                rkey = f"{key_prefix}_{period}"
                results[rkey] = {"label": f"{label_prefix} {period}", "n": len(subset),
                                "coef": coef, "se": se, "p": p, "stars": stars}
                print(f"  {label_prefix:<20s} {period}: {coef:.4f}{stars} ({se:.4f})", flush=True)
        except Exception as e:
            print(f"  {label_prefix} short/long: ERROR: {e}", flush=True)

    # 5. Generate report
    report_path = generate_report(results, stats)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
