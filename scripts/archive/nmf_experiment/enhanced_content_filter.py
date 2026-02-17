"""
enhanced_content_filter.py

Robustness: Re-aggregate newspaper slant after applying enhanced content filters
inspired by Widmer et al. (2025):
  1. Widmer-style text-level keyword filters (funeral home, hospital births, etc.)
  2. Wire article detection (AP, Reuters, UPI dateline patterns)

Two-phase approach for memory efficiency:
  Phase 1: Scan raw yearly parquets → save boolean flags per year (no text in memory)
  Phase 2: Load flags → match to congress data → re-aggregate → analyze → report
"""

import gc
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

sys.path.insert(0, os.path.join(os.environ["SHIFTING_SLANT_DIR"], "scripts", "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
RAW_YEARLY_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "yearly"
LABEL_DIR = cfg.NEWSPAPER_LABELS
NEWS_DIR = cfg.NEWS_DIR
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FLAG_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "text_flags"
FLAG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = BASE_DIR / "reports" / "robustness" / "enhanced_filter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAFTA_YEAR = 1994
BASE_YEAR = 1993
END_YEAR = 2004

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

# ── Widmer-style text-level keyword patterns (Table B.6) ───────────
WIDMER_TEXT_PATTERNS = [
    r"funeral service", r"funeral home", r"funeral notice",
    r"listing of deaths", r"local deaths",
    r"all deaths below were recorded",
    r"listing of births", r"local births", r"hospital births",
    r"all births below were recorded",
    r"these births were recorded at",
    r"birth announcement",
    r"wedding announcement",
    r"hospital notes",
]
WIDMER_TEXT_RE = re.compile("|".join(WIDMER_TEXT_PATTERNS), re.IGNORECASE)

WIDMER_START_PATTERNS = [
    r"^births\b", r"^deaths\b",
    r"^rcal-births", r"^local births", r"^local deaths",
]
WIDMER_START_RE = re.compile("|".join(WIDMER_START_PATTERNS), re.IGNORECASE)

# ── Wire article detection patterns ────────────────────────────────
WIRE_PATTERNS = [
    r"\(AP\)", r"\(Reuters\)", r"\(UPI\)",
    r"Associated Press",
    r"— AP\b", r"-- AP\b", r"—AP\b", r"--AP\b",
    r"— Reuters\b", r"-- Reuters\b",
    r"— UPI\b", r"-- UPI\b",
    r"By The Associated Press", r"By Associated Press",
    r"\bAP Wire\b",
]
WIRE_RE = re.compile("|".join(WIRE_PATTERNS), re.IGNORECASE)


# ==================================================================
# Phase 1: Precompute text-level flags per year
# ==================================================================
def phase1_compute_flags():
    """Scan raw text one year at a time, save lightweight flag files."""
    print("=" * 72)
    print("PHASE 1: Computing text-level exclusion flags")
    print("=" * 72)

    total_widmer = 0
    total_wire = 0
    total_articles = 0

    for year in range(1987, 2005):
        flag_path = FLAG_DIR / f"text_flags_{year}.parquet"
        raw_path = RAW_YEARLY_DIR / f"newspapers_{year}.parquet"
        if not raw_path.exists():
            print(f"  {year}: SKIPPED (no raw data)", flush=True)
            continue
        if flag_path.exists():
            print(f"  {year}: SKIPPED (flags exist)", flush=True)
            continue

        # Load only text column + matching keys
        df = pd.read_parquet(raw_path, columns=["paper", "title", "word_count", "text"])
        n = len(df)
        text = df["text"].fillna("")

        # Apply Widmer text patterns (search full text)
        widmer_match = text.str.contains(WIDMER_TEXT_RE, na=False)
        # Start-of-text patterns (first 80 chars only)
        widmer_start = text.str[:80].str.contains(WIDMER_START_RE, na=False)
        exclude_widmer = widmer_match | widmer_start

        # Wire detection
        exclude_wire = text.str.contains(WIRE_RE, na=False)

        # Save flags (NO text - just keys + booleans)
        flags = pd.DataFrame({
            "paper": df["paper"],
            "title": df["title"],
            "word_count": df["word_count"],
            "exclude_widmer": exclude_widmer,
            "exclude_wire": exclude_wire,
        })
        flags.to_parquet(flag_path, index=False)

        nw = exclude_widmer.sum()
        nwr = exclude_wire.sum()
        total_widmer += nw
        total_wire += nwr
        total_articles += n

        print(f"  {year}: {n:>9,} articles | "
              f"Widmer: {nw:>7,} ({nw/n*100:.1f}%) | "
              f"Wire: {nwr:>7,} ({nwr/n*100:.1f}%)", flush=True)

        del df, text, flags, widmer_match, widmer_start, exclude_widmer, exclude_wire
        gc.collect()

    print(f"\n  TOTAL: {total_articles:,} articles | "
          f"Widmer: {total_widmer:,} ({total_widmer/total_articles*100:.1f}%) | "
          f"Wire: {total_wire:,} ({total_wire/total_articles*100:.1f}%)")


# ==================================================================
# Phase 2: Match flags to congress data, re-aggregate, analyze
# ==================================================================
def phase2_analyze():
    """Load precomputed flags, match to congress articles, re-aggregate, run regressions."""

    congresses = cfg.get_congresses()

    # ── Load year-level flags into memory (lightweight, no text) ───
    print("\n" + "=" * 72)
    print("PHASE 2: Matching flags and re-aggregating")
    print("=" * 72)

    yearly_flags = {}
    for year in range(1987, 2005):
        flag_path = FLAG_DIR / f"text_flags_{year}.parquet"
        if flag_path.exists():
            yearly_flags[year] = pd.read_parquet(flag_path)

    # ── Build exclusion masks per congress ─────────────────────────
    print("\n  Building per-congress exclusion masks...")
    exclusion_by_cong = {}

    for cong in congresses:
        meta_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        if not meta_path.exists():
            continue

        meta = pd.read_parquet(meta_path, columns=["paper", "year", "title", "word_count"])

        sample_idx_path = NEWS_DIR / f"07_sample_idx_cong_{cong}.npy"
        if sample_idx_path.exists():
            idx = np.load(sample_idx_path)
            meta = meta.iloc[idx].reset_index(drop=True)

        n_articles = len(meta)
        exc_widmer = np.zeros(n_articles, dtype=bool)
        exc_wire = np.zeros(n_articles, dtype=bool)

        # Match year by year using positional alignment
        for year in sorted(meta["year"].unique()):
            if year not in yearly_flags:
                continue

            flags = yearly_flags[year]
            year_mask = meta["year"] == year
            year_meta = meta[year_mask].copy()
            year_idx = np.where(year_mask)[0]

            # Create composite key for matching
            year_meta["_key"] = (year_meta["paper"].astype(str) + "|" +
                                  year_meta["title"].fillna("").astype(str) + "|" +
                                  year_meta["word_count"].fillna(0).astype(int).astype(str))
            flags["_key"] = (flags["paper"].astype(str) + "|" +
                              flags["title"].fillna("").astype(str) + "|" +
                              flags["word_count"].fillna(0).astype(int).astype(str))

            # Build lookup: key -> (widmer, wire) flags (first occurrence)
            flag_lookup = flags.drop_duplicates(subset="_key", keep="first").set_index("_key")

            matched_widmer = year_meta["_key"].map(
                flag_lookup["exclude_widmer"]).fillna(False).values.astype(bool)
            matched_wire = year_meta["_key"].map(
                flag_lookup["exclude_wire"]).fillna(False).values.astype(bool)

            exc_widmer[year_idx] = matched_widmer
            exc_wire[year_idx] = matched_wire

        exclusion_by_cong[cong] = {"widmer": exc_widmer, "wire": exc_wire}
        nw = exc_widmer.sum()
        nwr = exc_wire.sum()
        print(f"    Congress {cong}: {n_articles:,} | "
              f"Widmer: {nw:,} ({nw/n_articles*100:.1f}%) | "
              f"Wire: {nwr:,} ({nwr/n_articles*100:.1f}%)")

    del yearly_flags
    gc.collect()

    # ── Re-aggregate with 3 filter configurations ──────────────────
    print("\n  Re-aggregating newspaper-year panels...")

    SLANT_COLS = ["right_intensity", "left_intensity", "net_slant", "politicization",
                  "right_norm", "left_norm", "net_slant_norm", "politicization_norm"]

    configs = {
        "baseline": "is_news only (no NMF)",
        "plus_widmer": "is_news + Widmer text keywords",
        "plus_wire": "is_news + Widmer text + wire removal",
    }
    panels = {}

    for config_name, config_desc in configs.items():
        chunks = []
        for cong in congresses:
            label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
            slant_path = NEWS_DIR / f"09_article_slant_norm_cong_{cong}.parquet"
            if not label_path.exists() or not slant_path.exists():
                continue

            meta = pd.read_parquet(label_path)
            slant = pd.read_parquet(slant_path)

            sample_idx_path = NEWS_DIR / f"07_sample_idx_cong_{cong}.npy"
            if sample_idx_path.exists():
                idx = np.load(sample_idx_path)
                meta = meta.iloc[idx].reset_index(drop=True)

            assert len(meta) == len(slant)
            df = pd.concat([meta, slant], axis=1)

            # Base: is_news only (no NMF)
            mask = df["is_news"].values.copy()

            # Additional filters
            if config_name in ("plus_widmer", "plus_wire"):
                if cong in exclusion_by_cong:
                    mask = mask & ~exclusion_by_cong[cong]["widmer"]
            if config_name == "plus_wire":
                if cong in exclusion_by_cong:
                    mask = mask & ~exclusion_by_cong[cong]["wire"]

            df = df[mask].copy()

            agg = (df.groupby(["paper", "year"])
                   .agg(n_articles=("net_slant", "count"),
                        **{col: (col, "mean") for col in SLANT_COLS})
                   .reset_index())
            grp = df.groupby(["paper", "year"])["net_slant"]
            agg["share_nonzero"] = grp.apply(lambda x: (x != 0).mean()).values
            agg["share_R"] = grp.apply(lambda x: (x > 0).mean()).values
            agg["share_D"] = grp.apply(lambda x: (x < 0).mean()).values
            chunks.append(agg)
            del meta, slant, df, agg
            gc.collect()

        panel = pd.concat(chunks, ignore_index=True).sort_values(["paper", "year"]).reset_index(drop=True)
        panels[config_name] = panel
        total = panel["n_articles"].sum()
        print(f"    {config_name}: {len(panel):,} rows, {panel['paper'].nunique()} papers, "
              f"{total:,} articles, share_R={panel['share_R'].mean():.4f}")

    # ── Merge with econ data and run regressions ───────────────────
    print("\n  Running regressions...")
    base_panel = pd.read_parquet(PANEL_PATH)
    base_panel["state_fips"] = base_panel["fips"] // 1000
    econ_cols = ["paper", "year", "cz", "state_fips", "vulnerability1990_scaled",
                 "manushare1990", "china_shock"]
    econ = base_panel[econ_cols].drop_duplicates(subset=["paper", "year"])

    results = {}
    for config_name, panel in panels.items():
        df = panel.merge(econ, on=["paper", "year"], how="inner")
        df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
        df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
        df = df[df["division"].notna()].copy()
        df["division"] = df["division"].astype(int)
        df = df[(df["year"] >= 1987) & (df["year"] <= END_YEAR)].copy()

        df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
        df["vuln_post"] = df["vulnerability1990_scaled"] * df["post"]

        year_list = sorted([y for y in df["year"].unique() if y != BASE_YEAR])
        for y in year_list:
            df[f"vuln_y{y}"] = df["vulnerability1990_scaled"] * (df["year"] == y).astype(int)
            df[f"manu_y{y}"] = df["manushare1990"].fillna(0) * (df["year"] == y).astype(int)
            df[f"china_y{y}"] = df["china_shock"].fillna(0) * (df["year"] == y).astype(int)
        df["div_year"] = df["division"].astype(str) + "_" + df["year"].astype(str)

        # Event study
        vuln_vars = " + ".join([f"vuln_y{y}" for y in year_list])
        ctrl_vars = " + ".join([f"manu_y{y} + china_y{y}" for y in year_list])
        fml = f"share_R ~ {vuln_vars} + {ctrl_vars} | paper + year + div_year"
        try:
            m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
            es_coefs = {y: {"coef": m.coef()[f"vuln_y{y}"], "se": m.se()[f"vuln_y{y}"]}
                        for y in year_list}
        except Exception as e:
            print(f"    Event study failed for {config_name}: {e}")
            es_coefs = {}

        # DiD (3 specs)
        did_results = []
        specs = [
            f"share_R ~ vuln_post | paper + year + div_year",
            f"share_R ~ vuln_post + {' + '.join([f'manu_y{y}' for y in year_list])} | paper + year + div_year",
            f"share_R ~ vuln_post + {ctrl_vars} | paper + year + div_year",
        ]
        for fml_did in specs:
            try:
                m_did = pf.feols(fml_did, data=df, vcov={"CRV1": "cz"})
                did_results.append({
                    "coef": m_did.coef()["vuln_post"],
                    "se": m_did.se()["vuln_post"],
                    "pval": m_did.pvalue()["vuln_post"],
                    "n": m_did._N,
                })
            except Exception as e:
                print(f"    DiD failed: {e}")
                did_results.append({"coef": np.nan, "se": np.nan, "pval": np.nan, "n": 0})

        results[config_name] = {"event_study": es_coefs, "did": did_results,
                                 "n_obs": len(df), "n_papers": df["paper"].nunique()}
        d = did_results[-1]
        print(f"    {config_name}: DiD Col3 = {d['coef']:.3f} ({d['se']:.3f}), p={d['pval']:.4f}")

    return results, panels


# ==================================================================
# Figures
# ==================================================================
def make_figures(results):
    colors = {"baseline": "#2d2d2d", "plus_widmer": "#bf6b63", "plus_wire": "#4a8c6f"}
    markers = {"baseline": "o", "plus_widmer": "s", "plus_wire": "D"}
    labels = {"baseline": "is\_news only",
              "plus_widmer": "+ Widmer text keywords",
              "plus_wire": "+ Widmer + wire removal"}
    offsets = {"baseline": -0.15, "plus_widmer": 0, "plus_wire": 0.15}

    # Event study comparison (paper style)
    fig, ax = plt.subplots(figsize=(12, 6))
    for cn in ["baseline", "plus_widmer", "plus_wire"]:
        es = results[cn]["event_study"]
        if not es:
            continue
        # Insert base year at zero
        all_years = sorted(list(es.keys()) + [BASE_YEAR])
        coefs = [es[y]["coef"] if y != BASE_YEAR else 0.0 for y in all_years]
        ses = [es[y]["se"] if y != BASE_YEAR else 0.0 for y in all_years]
        ci_lo = [c - 1.96*s for c, s in zip(coefs, ses)]
        ci_hi = [c + 1.96*s for c, s in zip(coefs, ses)]
        x = np.array([y + offsets[cn] for y in all_years])
        ax.errorbar(x, coefs,
                    yerr=[np.array(coefs) - np.array(ci_lo),
                          np.array(ci_hi) - np.array(coefs)],
                    fmt=markers[cn], color=colors[cn], markersize=4, capsize=2.5,
                    linewidth=1.0, label=labels[cn])
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.7, label="NAFTA (1994)")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    all_yrs = sorted(results["baseline"]["event_study"].keys()) + [BASE_YEAR]
    all_yrs = sorted(set(all_yrs))
    ax.set_xticks([yr for yr in all_yrs if yr % 2 == 1 or yr == all_yrs[0]])
    ax.set_xticklabels([str(yr) for yr in all_yrs if yr % 2 == 1 or yr == all_yrs[0]], fontsize=10)
    ax.set_xlim(all_yrs[0] - 0.7, all_yrs[-1] + 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "event_study_comparison.pdf", dpi=200,
                bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"  Saved: event_study_comparison.pdf")

    # DiD bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(3)
    width = 0.25
    spec_labels = ["(1) FE only", "(2) + manu", "(3) + manu + china"]
    for i, cn in enumerate(["baseline", "plus_widmer", "plus_wire"]):
        did = results[cn]["did"]
        coefs = [d["coef"] for d in did]
        ses = [d["se"] for d in did]
        ax.bar(x_pos + i*width, coefs, width, yerr=[1.96*s for s in ses],
               color=colors[cn], alpha=0.85, label=labels[cn], capsize=3)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(spec_labels, fontsize=10)
    ax.set_ylabel(r"DiD coefficient ($\beta$)")
    ax.set_title("DiD Coefficients: Enhanced Content Filtering Robustness")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "did_comparison.pdf", dpi=300)
    plt.close(fig)
    print(f"  Saved: did_comparison.pdf")


# ==================================================================
# LaTeX report
# ==================================================================
def make_report(results, panels):
    base_art = panels["baseline"]["n_articles"].sum()
    widmer_art = panels["plus_widmer"]["n_articles"].sum()
    wire_art = panels["plus_wire"]["n_articles"].sum()
    widmer_rm = base_art - widmer_art
    wire_rm = widmer_art - wire_art
    total_rm = base_art - wire_art

    def fmt_did(d):
        stars = "***" if d["pval"] < 0.01 else "**" if d["pval"] < 0.05 else "*" if d["pval"] < 0.1 else ""
        return f"${d['coef']:.3f}{stars}$", f"$({d['se']:.3f})$"

    r = results
    d = [[fmt_did(r[cn]["did"][s]) for cn in ["baseline","plus_widmer","plus_wire"]] for s in range(3)]

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{mathpazo}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{threeparttable}
\hypersetup{colorlinks=true, linkcolor=blue!60!black, citecolor=blue!60!black}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\title{\textbf{Enhanced Content Filtering Robustness}\\[8pt]
\large Widmer-Style Text Filters and Wire Article Removal}
\author{Research Report}
\date{February 2026}

\begin{document}
\maketitle

\section{Motivation}

Widmer et al.\ (2025), using the same NewsLibrary data, apply content filters including text-level routine announcement keywords (``funeral home,'' ``hospital births,'' etc., Table B.6) and wire article removal. This report tests whether applying Widmer-style filters to our pipeline (without NMF topic filtering) changes the main results. The baseline here uses only our title-based \texttt{is\_news} filter.

\section{Filter Definitions}

\textbf{Widmer text keywords.} Following Widmer et al.\ (2025, Table B.6), we search article \emph{text} (not just titles) for: ``funeral service/home/notice,'' ``listing of births/deaths,'' ``hospital births/notes,'' ``birth/wedding announcements,'' and start-of-text patterns like ``births'' or ``deaths.''

\textbf{Wire article detection.} We flag articles whose text contains wire service identifiers: ``(AP),'' ``(Reuters),'' ``(UPI),'' ``Associated Press,'' or related byline/dateline patterns.

\section{Filter Impact}

""" + f"""\\begin{{table}}[H]
\\centering
\\caption{{Article Counts by Filter Configuration}}
\\begin{{tabular}}{{@{{}} l r r r @{{}}}}
\\toprule
& Baseline & + Widmer Text & + Wire Removal \\\\
\\midrule
Total articles & {base_art:,} & {widmer_art:,} & {wire_art:,} \\\\
Additional removed & --- & {widmer_rm:,} & {wire_rm:,} \\\\
\\% of baseline & 100\\% & {widmer_art/base_art*100:.1f}\\% & {wire_art/base_art*100:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Event Study}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=\\textwidth]{{event_study_comparison.pdf}}
    \\caption{{Event study coefficients (vulnerability $\\times$ year) for share\\_R under three filter configurations. Shaded bands: 95\\% CI. All specifications include paper FE, year FE, division$\\times$year FE, year-specific manufacturing share and China shock controls.}}
\\end{{figure}}

\\section{{DiD Results}}

\\begin{{table}}[H]
\\centering
\\caption{{DiD Coefficients by Filter Configuration}}
\\small
\\begin{{threeparttable}}
\\begin{{tabular}}{{@{{}} l cc cc cc @{{}}}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Baseline}} & \\multicolumn{{2}}{{c}}{{+ Widmer Text}} & \\multicolumn{{2}}{{c}}{{+ Wire Removal}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
(1) FE only & {d[0][0][0]} & & {d[0][1][0]} & & {d[0][2][0]} & \\\\
& {d[0][0][1]} & & {d[0][1][1]} & & {d[0][2][1]} & \\\\
\\addlinespace
(2) + manu & {d[1][0][0]} & & {d[1][1][0]} & & {d[1][2][0]} & \\\\
& {d[1][0][1]} & & {d[1][1][1]} & & {d[1][2][1]} & \\\\
\\addlinespace
(3) + manu + china & {d[2][0][0]} & & {d[2][1][0]} & & {d[2][2][0]} & \\\\
& {d[2][0][1]} & & {d[2][1][1]} & & {d[2][2][1]} & \\\\
\\addlinespace
N & {r['baseline']['did'][2]['n']:,} & & {r['plus_widmer']['did'][2]['n']:,} & & {r['plus_wire']['did'][2]['n']:,} & \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\footnotesize
\\item \\textit{{Notes:}} Each cell reports the DiD coefficient on vulnerability$\\times$post. All regressions include paper FE, year FE, division$\\times$year FE. SEs clustered by CZ. $^{{*}}$~$p<0.10$, $^{{**}}$~$p<0.05$, $^{{***}}$~$p<0.01$.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.85\\textwidth]{{did_comparison.pdf}}
    \\caption{{DiD coefficients across three specifications and filter configurations. Error bars: 95\\% CI.}}
\\end{{figure}}

\\section{{Conclusion}}

Without NMF topic filtering, the Widmer-style text keyword filters and wire article removal serve as alternative content cleaning approaches. The results show whether these filters meaningfully affect the estimated effect of NAFTA vulnerability on newspaper slant.

\\end{{document}}
"""

    tex_path = OUT_DIR / "enhanced_filter_robustness.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Saved: {tex_path}")
    return tex_path


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    t0 = time.time()

    # Phase 1: compute flags (skip if ALL years done)
    all_flags_exist = all((FLAG_DIR / f"text_flags_{y}.parquet").exists()
                          for y in range(1987, 2005))
    if not all_flags_exist:
        phase1_compute_flags()
    else:
        print("Phase 1: All flags already computed, skipping.")

    # Phase 2: analyze
    results, panels = phase2_analyze()

    # Figures
    print("\n" + "=" * 72)
    print("FIGURES")
    print("=" * 72)
    make_figures(results)

    # Report
    print("\n" + "=" * 72)
    print("REPORT")
    print("=" * 72)
    tex_path = make_report(results, panels)

    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Report: {tex_path}")
    print(f"{'=' * 72}")
