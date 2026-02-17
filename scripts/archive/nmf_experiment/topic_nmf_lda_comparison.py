"""
Topic decomposition robustness: NMF and LDA comparison.

Replaces keyword-based topic classification with data-driven topic discovery
(NMF and LDA), then re-runs the same DiD regressions to test whether the
"diffuse effect" finding is robust to topic classification method.

Strategy for memory efficiency:
- Fit NMF/LDA on a random subsample (~500K articles)
- Transform each congress separately using the fitted model
- Aggregate to newspaper-year level per congress, then combine

Outputs:
  - output/tables/topic_nmf_results.csv
  - output/tables/topic_lda_results.csv
  - output/tables/topic_method_comparison.csv
  - output/figures/topic_method_comparison.pdf
  - output/figures/topic_method_summary.pdf
"""

import os, sys, gc, time, warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyfixest as pf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
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
FIG_DIR = cfg.FIG_DIR
TAB_DIR = cfg.TAB_DIR

NAFTA_YEAR = 1994
END_YEAR = 2004
N_TOPICS = 10
RANDOM_STATE = 42
SUBSAMPLE_SIZE = 500_000  # Fit models on this many articles

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
    """Load shared vocabulary."""
    print("Loading vocabulary ...", flush=True)
    vec = joblib.load(SPEECH_DIR / "05_vectorizer.joblib")
    all_features = vec.get_feature_names_out()

    shared_vocab_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    shared_vocab_mask = np.load(shared_vocab_path) if shared_vocab_path.exists() else None
    if shared_vocab_mask is not None:
        features = all_features[shared_vocab_mask]
    else:
        features = all_features
        shared_vocab_mask = None
    print(f"  {len(features)} features", flush=True)
    return features, shared_vocab_mask


def load_congress_news(cong, shared_vocab_mask):
    """Load feature matrix and metadata for one congress, filtered to news."""
    label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
    if not label_path.exists():
        return None, None
    meta = pd.read_parquet(label_path, columns=["paper", "year", "is_news"])

    sample_idx_path = NEWS_FEATURES_DIR / f"07_sample_idx_cong_{cong}.npy"
    if sample_idx_path.exists():
        idx = np.load(sample_idx_path)
        meta = meta.iloc[idx].reset_index(drop=True)

    slant_path = NEWS_DIR / f"08_article_slant_cong_{cong}.parquet"
    if not slant_path.exists():
        return None, None
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
    return X_news, meta_news


def build_training_sample(shared_vocab_mask, subsample_size):
    """Build a random subsample across congresses for fitting topic models."""
    print(f"\nBuilding training sample ({subsample_size:,} articles) ...", flush=True)
    congresses = [w[-1] for w in cfg.get_windows()]
    rng = np.random.default_rng(RANDOM_STATE)

    # First pass: count articles per congress
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
    print(f"  Total news articles across congresses: {total:,}", flush=True)

    # Proportional sampling from each congress
    samples = []
    for cong in congresses:
        if cong not in counts:
            continue
        n_from_cong = int(subsample_size * counts[cong] / total)
        if n_from_cong < 100:
            n_from_cong = min(100, counts[cong])

        X_news, _ = load_congress_news(cong, shared_vocab_mask)
        if X_news is None:
            continue

        n = X_news.shape[0]
        if n_from_cong >= n:
            samples.append(X_news)
        else:
            idx = rng.choice(n, size=n_from_cong, replace=False)
            idx.sort()
            samples.append(X_news[idx])

        print(f"    Congress {cong}: sampled {samples[-1].shape[0]:,} / {n:,}", flush=True)
        del X_news
        gc.collect()

    X_train = sp.vstack(samples, format="csr")
    print(f"  Training sample: {X_train.shape[0]:,} articles, {X_train.shape[1]:,} features", flush=True)
    del samples
    gc.collect()
    return X_train


def fit_nmf(X_train, n_topics, features):
    """Fit NMF on training sample."""
    print(f"\nFitting NMF with {n_topics} topics ...", flush=True)
    t0 = time.time()
    model = NMF(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=300,
        init="nndsvda",
    )
    model.fit(X_train)
    elapsed = time.time() - t0
    print(f"  NMF fit in {elapsed:.1f}s, reconstruction error: {model.reconstruction_err_:.2f}", flush=True)

    H = model.components_
    print(f"\n  NMF Topics (top 10 words):", flush=True)
    for k in range(n_topics):
        top_idx = H[k].argsort()[::-1][:10]
        top_words = [features[i] for i in top_idx]
        print(f"    Topic {k+1}: {', '.join(top_words)}", flush=True)

    return model


def fit_lda(X_train, n_topics, features):
    """Fit LDA on training sample."""
    print(f"\nFitting LDA with {n_topics} topics ...", flush=True)
    t0 = time.time()
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=30,
        learning_method="online",
        batch_size=4096,
        n_jobs=-1,
    )
    model.fit(X_train)
    elapsed = time.time() - t0
    print(f"  LDA fit in {elapsed:.1f}s", flush=True)

    H = model.components_
    print(f"\n  LDA Topics (top 10 words):", flush=True)
    for k in range(n_topics):
        top_idx = H[k].argsort()[::-1][:10]
        top_words = [features[i] for i in top_idx]
        print(f"    Topic {k+1}: {', '.join(top_words)}", flush=True)

    return model


def label_topics(H, features):
    """Auto-label topics by matching top words to keyword categories.

    Uses startswith matching (not substring) to avoid false positives
    like "day" matching "yesterday". Includes both content-type categories
    (for NMF/LDA-discovered topics) and policy categories.
    """
    label_keywords = {
        # Content types (discovered by NMF/LDA)
        "Sports": ["game", "team", "season", "play", "win", "coach", "point", "yard", "goal", "league"],
        "Obituaries": ["die", "funer", "church", "born", "servic", "memori", "surviv"],
        "Crime & police": ["polic", "arrest", "charg", "offic", "investig", "suspect", "shoot", "victim"],
        "Local government": ["citi", "council", "mayor", "park", "zoning", "approv"],
        "County & judicial": ["counti", "court", "judg", "sheriff", "commiss", "clerk"],
        "National & business": ["presid", "compani", "york", "administr", "corpor", "firm", "busi"],
        "Business & economy": ["million", "percent", "market", "profit", "econom", "stock", "revenue"],
        "Arts & culture": ["music", "art", "perform", "film", "theater", "museum", "concert", "festiv"],
        "Spanish-language": ["el", "en", "lo", "su", "contra", "por", "una", "las", "cuba"],
        "Government & institutions": ["state", "senat", "depart", "govern", "feder", "agenc", "univers", "colleg"],
        "General features": ["ago", "life", "stori", "live", "famili", "home", "world"],
        "Human interest": ["peopl", "say", "like", "just", "know", "think", "want", "feel"],
        "Politics & elections": ["elect", "vote", "democrat", "republican", "campaign", "poll", "candid"],
        "National & defense": ["war", "nation", "forc", "militari", "troop", "secur", "attack", "foreign"],
        "Congressional speech": ["honor", "behalf", "uniform", "recogn", "gentleman", "yield", "commend"],
        # Policy categories (for keyword-based topics)
        "Tax & fiscal": ["tax", "budget", "spend", "deficit", "fiscal", "appropriat"],
        "Healthcare": ["health", "medicar", "insur", "hospit", "patient", "medicaid"],
        "Social policy": ["welfar", "social", "child", "pension", "retir"],
        "Defense & foreign": ["militari", "defens", "troop", "foreign", "terror"],
        "Trade": ["trade", "tariff", "nafta", "export", "import"],
        "Environment & energy": ["environ", "energi", "oil", "pollut", "climat"],
        "Education": ["educ", "school", "student", "teacher", "colleg", "univers"],
        "Crime & justice": ["crime", "gun", "prison", "victim", "crimin"],
        "Government & procedure": ["bill", "committe", "legisl", "amendment"],
        "Civil rights & identity": ["civil", "right", "women", "race", "immigr", "abort"],
    }

    n_topics = H.shape[0]
    labels = []
    used = set()

    for k in range(n_topics):
        top_idx = H[k].argsort()[::-1][:20]
        top_words = [features[i] for i in top_idx]

        best_label, best_score = None, 0
        for cat, kws in label_keywords.items():
            if cat in used:
                continue
            # Use startswith matching to avoid false positives
            score = sum(1 for w in top_words for kw in kws if w.startswith(kw))
            if score > best_score:
                best_score = score
                best_label = cat

        if best_label and best_score >= 2:
            labels.append(best_label)
            used.add(best_label)
        else:
            labels.append(f"Topic {k+1}")

    return labels


def process_congress_with_model(cong, model, shared_vocab_mask, n_topics):
    """Transform one congress's articles and compute newspaper-year aggregates.

    Uses DOMINANT TOPIC assignment: each article assigned to its highest-weight
    topic. This avoids threshold sensitivity and ensures every article belongs
    to exactly one topic (no residual "no specific topic" category).
    """
    X_news, meta = load_congress_news(cong, shared_vocab_mask)
    if X_news is None or meta is None:
        return None

    n_articles = X_news.shape[0]
    is_R = (meta["net_slant"].values > 0).astype(np.int8)

    # Transform: get document-topic weights
    W = model.transform(X_news)  # (n_articles, n_topics)
    del X_news

    # Dominant topic: each article → its highest-weight topic
    dominant = W.argmax(axis=1)  # (n_articles,)
    del W

    # Build article-level dataframe
    df_art = pd.DataFrame({
        "paper": meta["paper"].values,
        "year": meta["year"].values,
        "is_R": is_R,
    })

    for k in range(n_topics):
        flag = (dominant == k).astype(np.int8)
        df_art[f"topic_{k}"] = flag
        df_art[f"R_in_{k}"] = is_R * flag

    # Aggregate
    grp = df_art.groupby(["paper", "year"])
    agg = grp.agg(
        n_articles=("is_R", "count"),
        n_R=("is_R", "sum"),
        share_R=("is_R", "mean"),
    ).reset_index()

    for k in range(n_topics):
        agg[f"n_topic_{k}"] = grp[f"topic_{k}"].sum().values
        agg[f"nR_topic_{k}"] = grp[f"R_in_{k}"].sum().values
        agg[f"share_R_topic_{k}"] = (
            agg[f"nR_topic_{k}"] / agg[f"n_topic_{k}"]
        ).replace([np.inf, -np.inf], np.nan)

    return agg


def run_did_regressions(panel_topic, topic_labels, method_name):
    """Merge with regression panel and run DiD for each topic."""
    print(f"\n{'='*70}", flush=True)
    print(f"  DiD Regressions: {method_name}", flush=True)
    print(f"{'='*70}", flush=True)

    reg = pd.read_parquet(PANEL_PATH)
    reg = reg[reg["cz"].notna() & reg["vulnerability1990_scaled"].notna()].copy()
    reg = reg[reg["year"] <= END_YEAR].copy()

    merged = panel_topic.merge(
        reg[["paper", "year", "cz", "fips", "vulnerability1990_scaled",
             "manushare1990", "china_shock"]],
        on=["paper", "year"], how="inner"
    )
    print(f"  Merged: {len(merged)} obs", flush=True)

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

    n_topics = len(topic_labels)
    outcome_list = [("share_R", "All articles", "all")]
    for k in range(n_topics):
        outcome_list.append((f"share_R_topic_{k}", topic_labels[k], f"topic_{k}"))

    results = []
    print(f"\n  {'Outcome':<30s} {'N':>6s} {'Coef':>8s} {'SE':>8s} {'p':>8s} {'Share':>6s}", flush=True)
    print("  " + "-" * 72, flush=True)

    for depvar, label, key in outcome_list:
        if depvar not in merged.columns:
            continue
        subset = merged[merged[depvar].notna()].copy()
        n_obs = len(subset)
        if n_obs < 100:
            continue

        fml = f"{depvar} ~ {fml_base}"
        try:
            m = pf.feols(fml, data=subset, vcov={"CRV1": "cz"})
            t = m.tidy().loc["vuln_x_post"]
            coef, se, p = t["Estimate"], t["Std. Error"], t["Pr(>|t|)"]
        except Exception as e:
            print(f"  {label:<30s}  ERROR: {e}", flush=True)
            continue

        if key == "all":
            avg_share = 1.0
        else:
            n_col = f"n_{key}"
            avg_share = merged[n_col].sum() / merged["n_articles"].sum() if n_col in merged.columns else np.nan

        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {label:<30s} {n_obs:>6d} {coef:>8.4f}{stars:3s} {se:>8.4f} {p:>8.4f} {avg_share:>6.1%}", flush=True)

        results.append({
            "method": method_name,
            "topic": label,
            "key": key,
            "n_obs": n_obs,
            "coef": coef,
            "se": se,
            "p_value": p,
            "avg_article_share": avg_share,
        })

    return pd.DataFrame(results)


def plot_comparison(keyword_df, nmf_df, lda_df):
    """Three-panel comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)

    datasets = [
        (keyword_df, "Keyword-based", axes[0]),
        (nmf_df, "NMF", axes[1]),
        (lda_df, "LDA", axes[2]),
    ]

    SIG_COLOR = "#2d2d2d"
    INSIG_COLOR = "#b0b0b0"
    REF_COLOR = "#7a7a7a"

    for df, title, ax in datasets:
        plot_data = df[df["key"] != "all"].copy()
        plot_data = plot_data.sort_values("coef", ascending=True).reset_index(drop=True)
        all_row = df[df["key"] == "all"]
        all_coef = all_row["coef"].values[0] if len(all_row) > 0 else 0

        y_pos = np.arange(len(plot_data))
        coefs = plot_data["coef"].values
        ses = plot_data["se"].values
        pvals = plot_data["p_value"].values

        for y in y_pos:
            ax.axhline(y, color="#f0f0f0", linewidth=0.6, zorder=0)

        ax.axvline(all_coef, color=REF_COLOR, linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
        ax.axvline(0, color="black", linewidth=0.6, zorder=1)

        for i, y in enumerate(y_pos):
            sig = pvals[i] < 0.05
            color = SIG_COLOR if sig else INSIG_COLOR
            ax.errorbar(coefs[i], y, xerr=1.96 * ses[i],
                        fmt="o", color=color, markersize=6,
                        capsize=3, capthick=1.2,
                        ecolor=color, elinewidth=1.2,
                        markeredgecolor="white", markeredgewidth=0.6,
                        alpha=0.9, zorder=3)

            share = plot_data["avg_article_share"].values[i]
            ax.text(0.24, y, f"{share:.0%}",
                    fontsize=6.5, color="#888888", va="center", ha="right")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_data["topic"], fontsize=7.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("DiD coefficient", fontsize=9)
        ax.set_xlim(-0.12, 0.25)

        n_sig = (pvals < 0.05).sum()
        n_pos = (coefs > 0).sum()
        ax.text(0.95, 0.02, f"{n_pos}/{len(coefs)} positive\n{n_sig}/{len(coefs)} sig.",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                color="#666666")

        ax.text(0.24, len(y_pos) - 0.3, "share", fontsize=6.5,
                color="#999999", va="bottom", ha="right", fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SIG_COLOR,
               markersize=7, markeredgecolor="white", label="$p < 0.05$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INSIG_COLOR,
               markersize=6, markeredgecolor="white", alpha=0.7, label="Not sig."),
        Line2D([0], [0], color=REF_COLOR, linestyle="--", linewidth=1.0,
               alpha=0.7, label="Overall effect"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=9, framealpha=0.95, edgecolor="#dddddd",
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_summary(keyword_df, nmf_df, lda_df):
    """Summary: topic coefficients by method (horizontal strip plot)."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    methods = ["Keyword", "NMF", "LDA"]
    datasets = [keyword_df, nmf_df, lda_df]
    colors = ["#2d2d2d", "#1b7837", "#762a83"]

    for i, (df, method, color) in enumerate(zip(datasets, methods, colors)):
        topics = df[~df["key"].isin(["all"])].copy()
        coefs = topics["coef"].values
        ses = topics["se"].values

        # Jitter y positions
        y_offsets = np.linspace(-0.2, 0.2, len(coefs))
        for j, (c, se) in enumerate(zip(coefs, ses)):
            ax.errorbar(c, i + y_offsets[j], xerr=1.96 * se,
                        fmt="o", color=color, markersize=3.5,
                        capsize=1.5, capthick=0.6,
                        ecolor=color, elinewidth=0.6,
                        markeredgecolor="white", markeredgewidth=0.3,
                        alpha=0.5, zorder=3)

        # Diamond for mean
        mean_coef = coefs.mean()
        ax.plot(mean_coef, i, "D", color=color, markersize=9,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)

        # Overall effect
        all_row = df[df["key"] == "all"]
        if len(all_row) > 0:
            ax.plot(all_row["coef"].values[0], i, "s", color=color,
                    markersize=7, markeredgecolor="white",
                    markeredgewidth=0.8, alpha=0.8, zorder=4)

    ax.axvline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel("DiD coefficient on Share R-leaning", fontsize=10)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=5, alpha=0.5, label="Individual topics"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#888888",
               markersize=8, markeredgecolor="white", label="Mean across topics"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888",
               markersize=6, markeredgecolor="white", label="Overall effect"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left", framealpha=0.95)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()

    features, shared_vocab_mask = load_vocab()
    congresses = [w[-1] for w in cfg.get_windows()]

    # ── Step 1: Build training sample ──
    X_train = build_training_sample(shared_vocab_mask, SUBSAMPLE_SIZE)

    # ── Step 2: Fit NMF ──
    nmf_model = fit_nmf(X_train, N_TOPICS, features)

    # ── Step 3: Fit LDA ──
    lda_model = fit_lda(X_train, N_TOPICS, features)

    del X_train
    gc.collect()

    # Auto-label topics
    nmf_labels = label_topics(nmf_model.components_, features)
    lda_labels = label_topics(lda_model.components_, features)
    print(f"\n  NMF labels: {nmf_labels}", flush=True)
    print(f"  LDA labels: {lda_labels}", flush=True)

    # ── Step 4: Process each congress and aggregate ──
    print("\nProcessing articles with NMF ...", flush=True)
    nmf_chunks = []
    for cong in congresses:
        t0 = time.time()
        agg = process_congress_with_model(cong, nmf_model, shared_vocab_mask, N_TOPICS)
        if agg is not None:
            nmf_chunks.append(agg)
            print(f"  Congress {cong}: {len(agg)} newspaper-year obs ({time.time()-t0:.1f}s)", flush=True)
        gc.collect()

    panel_nmf = pd.concat(nmf_chunks, ignore_index=True)
    panel_nmf = panel_nmf.groupby(["paper", "year"]).sum(numeric_only=True).reset_index()
    panel_nmf["share_R"] = panel_nmf["n_R"] / panel_nmf["n_articles"]
    for k in range(N_TOPICS):
        panel_nmf[f"share_R_topic_{k}"] = (
            panel_nmf[f"nR_topic_{k}"] / panel_nmf[f"n_topic_{k}"]
        ).replace([np.inf, -np.inf], np.nan)
    print(f"  NMF panel: {len(panel_nmf)} newspaper-year obs", flush=True)

    print("\nProcessing articles with LDA ...", flush=True)
    lda_chunks = []
    for cong in congresses:
        t0 = time.time()
        agg = process_congress_with_model(cong, lda_model, shared_vocab_mask, N_TOPICS)
        if agg is not None:
            lda_chunks.append(agg)
            print(f"  Congress {cong}: {len(agg)} newspaper-year obs ({time.time()-t0:.1f}s)", flush=True)
        gc.collect()

    panel_lda = pd.concat(lda_chunks, ignore_index=True)
    panel_lda = panel_lda.groupby(["paper", "year"]).sum(numeric_only=True).reset_index()
    panel_lda["share_R"] = panel_lda["n_R"] / panel_lda["n_articles"]
    for k in range(N_TOPICS):
        panel_lda[f"share_R_topic_{k}"] = (
            panel_lda[f"nR_topic_{k}"] / panel_lda[f"n_topic_{k}"]
        ).replace([np.inf, -np.inf], np.nan)
    print(f"  LDA panel: {len(panel_lda)} newspaper-year obs", flush=True)

    # ── Step 5: DiD regressions ──
    nmf_results = run_did_regressions(panel_nmf, nmf_labels, "NMF")
    lda_results = run_did_regressions(panel_lda, lda_labels, "LDA")

    # Save
    nmf_results.to_csv(TAB_DIR / "topic_nmf_results.csv", index=False, float_format="%.6f")
    lda_results.to_csv(TAB_DIR / "topic_lda_results.csv", index=False, float_format="%.6f")

    # ── Step 6: Save top words ──
    top_words_data = []
    for method, model, labels in [("NMF", nmf_model, nmf_labels), ("LDA", lda_model, lda_labels)]:
        H = model.components_
        for k in range(N_TOPICS):
            top_idx = H[k].argsort()[::-1][:15]
            top_words = [features[i] for i in top_idx]
            top_words_data.append({
                "method": method, "topic_idx": k, "label": labels[k],
                "top_words": ", ".join(top_words),
            })
    pd.DataFrame(top_words_data).to_csv(TAB_DIR / "topic_nmf_lda_top_words.csv", index=False)

    # ── Step 7: Load keyword results and plot ──
    keyword_csv = TAB_DIR / "topic_share_decomposition.csv"
    if keyword_csv.exists():
        keyword_results = pd.read_csv(keyword_csv)
        keyword_results["method"] = "Keyword"

        all_results = pd.concat([keyword_results, nmf_results, lda_results], ignore_index=True)
        all_results.to_csv(TAB_DIR / "topic_method_comparison.csv", index=False, float_format="%.6f")

        fig = plot_comparison(keyword_results, nmf_results, lda_results)
        fig.savefig(FIG_DIR / "topic_method_comparison.pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"\n  Comparison figure saved", flush=True)

        fig2 = plot_summary(keyword_results, nmf_results, lda_results)
        fig2.savefig(FIG_DIR / "topic_method_summary.pdf", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"  Summary figure saved", flush=True)

        # Summary
        print(f"\n{'='*70}", flush=True)
        print(f"  SUMMARY: Diffuse Effect Robustness", flush=True)
        print(f"{'='*70}", flush=True)
        for method in ["Keyword", "NMF", "LDA"]:
            sub = all_results[(all_results["method"] == method) & (all_results["key"] != "all")]
            if len(sub) == 0:
                continue
            n_pos = (sub["coef"] > 0).sum()
            n_sig = (sub["p_value"] < 0.05).sum()
            mean_c = sub["coef"].mean()
            overall = all_results[(all_results["method"] == method) & (all_results["key"] == "all")]["coef"].values
            overall = overall[0] if len(overall) > 0 else np.nan
            print(f"\n  {method}:", flush=True)
            print(f"    Overall: {overall:.4f}", flush=True)
            print(f"    Positive: {n_pos}/{len(sub)}, Significant: {n_sig}/{len(sub)}", flush=True)
            print(f"    Mean topic coef: {mean_c:.4f}", flush=True)
    else:
        print("  WARNING: Run topic_share_decomposition.py first for keyword baseline.", flush=True)

    total_time = time.time() - pipeline_start
    print(f"\nTotal time: {total_time:.1f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
