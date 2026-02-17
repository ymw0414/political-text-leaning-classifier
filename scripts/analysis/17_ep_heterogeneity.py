"""
17_ep_heterogeneity.py

Editor & Publisher data analysis:
  1. Circulation & ad price response to NAFTA (long-difference, 4-yr snapshots)
  2. Ad price response by initial partisan lean (demand-side evidence)

Inputs:
  - data/processed/panel/14_regression_panel.parquet
  - data/processed/panel/ep_newspaper_panel.parquet

Outputs:
  - Console output with results
"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
EP_PATH = BASE_DIR / "data" / "processed" / "panel" / "ep_newspaper_panel.parquet"

NAFTA_YEAR = 1994

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

# Papers to exclude (national, college, Spanish-language)
EXCLUDE_PAPERS = [
    'CHRISTIAN SCIENCE MONITOR',
    'USA TODAY',
    'Daily Pennsylvanian, The: University of Pennsylvania (Philadelphia, PA)',
    'Nuevo Herald, El (Miami, FL)',
]


def load_panel():
    """Load regression panel."""
    df = pd.read_parquet(PANEL_PATH)
    df = df[~df['paper'].isin(EXCLUDE_PAPERS)].copy()
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= 2004].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes
    df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

    return df


def part1_circulation_response(df):
    """Test whether circulation/ad price responds to NAFTA vulnerability (long difference)."""
    print("\n" + "=" * 70)
    print("  PART 1: Circulation & Ad Price Response to NAFTA")
    print("=" * 70)

    import statsmodels.api as sm

    ep = pd.read_parquet(EP_PATH)
    ep['ep_circ'] = pd.to_numeric(ep['ep_circ'], errors='coerce')
    ep['ep_adprice'] = pd.to_numeric(ep['ep_adprice'], errors='coerce')

    # Get paper-level vulnerability and controls
    paper_chars = df.groupby('paper').first()[
        ['vulnerability1990_scaled', 'manushare1990', 'cz', 'fips',
         'state_fips', 'division']
    ].reset_index()

    # Wide format: circ and adprice at each E&P year
    for var in ['ep_circ', 'ep_adprice']:
        wide = ep.pivot(index='paper', columns='ep_year', values=var).reset_index()
        wide.columns = ['paper'] + [f'{var}_{int(y)}' for y in wide.columns[1:]]
        paper_chars = paper_chars.merge(wide, on='paper', how='left')

    def stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    def run_long_diff(sub, var, end_yr):
        """Run a single long-difference regression, return dict or None."""
        col_base = f'{var}_1988'
        col_end = f'{var}_{end_yr}'
        if col_base not in sub.columns or col_end not in sub.columns:
            return None
        s = sub.dropna(subset=[col_base, col_end]).copy()
        if var == 'ep_circ':
            s['dep'] = (s[col_end] - s[col_base]) / s[col_base] * 100
        else:
            s['dep'] = s[col_end] - s[col_base]
        if len(s) < 5:
            return None
        X = sm.add_constant(s['vulnerability1990_scaled'])
        n_clusters = s['cz'].nunique()
        if n_clusters < 2:
            return None
        model = sm.OLS(s['dep'], X).fit(
            cov_type='cluster', cov_kwds={'groups': s['cz']})
        return {
            'coef': model.params['vulnerability1990_scaled'],
            'se': model.bse['vulnerability1990_scaled'],
            'p': model.pvalues['vulnerability1990_scaled'],
            'n': len(s),
            'mean_dep': s['dep'].mean(),
        }

    # --- 3A: Overall long differences ---
    print("\n  [1A] Overall Long Differences")
    for var, label in [('ep_circ', 'Circulation'), ('ep_adprice', 'Ad Price')]:
        print(f"\n  --- {label} ---")
        unit = "pp" if var == 'ep_circ' else "$/col-inch"
        for end_yr in [1996, 2000, 2004]:
            r = run_long_diff(paper_chars, var, end_yr)
            if r:
                print(f"    1988-{end_yr}: coef={r['coef']:>8.3f}{stars(r['p']):3s} "
                      f"(SE={r['se']:.3f}), N={r['n']}, "
                      f"mean dep={r['mean_dep']:.2f}{unit}")

def part2_ad_by_initial_lean(df):
    """Test whether ad price decline differs by initial partisan lean."""
    print("\n" + "=" * 70)
    print("  PART 2: Ad Price Response by Initial Partisan Lean")
    print("=" * 70)

    import statsmodels.api as sm

    # Compute pre-NAFTA mean Share R-leaning per newspaper
    pre = df[df['year'] < NAFTA_YEAR].groupby('paper')['share_R'].mean()
    pre.name = 'pre_nafta_share_R'
    median_lean = pre.median()
    print(f"\n  Pre-NAFTA Share R-leaning: median = {median_lean:.3f}")
    print(f"  D-leaning papers (below median): {(pre < median_lean).sum()}")
    print(f"  R-leaning papers (above median): {(pre >= median_lean).sum()}")

    # Load E&P data
    ep = pd.read_parquet(EP_PATH)
    ep['ep_adprice'] = pd.to_numeric(ep['ep_adprice'], errors='coerce')

    paper_chars = df.groupby('paper').first()[
        ['vulnerability1990_scaled', 'cz']
    ].reset_index()
    paper_chars = paper_chars.merge(pre.reset_index(), on='paper', how='left')
    paper_chars['initially_D'] = (paper_chars['pre_nafta_share_R'] < median_lean).astype(int)

    # Pivot ad prices wide
    wide = ep.pivot(index='paper', columns='ep_year', values='ep_adprice').reset_index()
    wide.columns = ['paper'] + [f'ep_adprice_{int(y)}' for y in wide.columns[1:]]
    paper_chars = paper_chars.merge(wide, on='paper', how='left')

    def stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    def run_long_diff(sub, end_yr):
        col_base = 'ep_adprice_1988'
        col_end = f'ep_adprice_{end_yr}'
        if col_base not in sub.columns or col_end not in sub.columns:
            return None
        s = sub.dropna(subset=[col_base, col_end]).copy()
        s['dep'] = s[col_end] - s[col_base]
        if len(s) < 5:
            return None
        X = sm.add_constant(s['vulnerability1990_scaled'])
        n_clusters = s['cz'].nunique()
        if n_clusters < 2:
            return None
        model = sm.OLS(s['dep'], X).fit(
            cov_type='cluster', cov_kwds={'groups': s['cz']})
        return {
            'coef': model.params['vulnerability1990_scaled'],
            'se': model.bse['vulnerability1990_scaled'],
            'p': model.pvalues['vulnerability1990_scaled'],
            'n': len(s),
        }

    print(f"\n  {'Period':<14} {'D-leaning':>30}   {'R-leaning':>30}")
    print(f"  {'':14s} {'coef (SE) N':>30}   {'coef (SE) N':>30}")
    print("  " + "-" * 76)

    for end_yr in [1996, 2000, 2004]:
        parts = []
        for label, mask in [('D-lean', paper_chars['initially_D'] == 1),
                             ('R-lean', paper_chars['initially_D'] == 0)]:
            r = run_long_diff(paper_chars[mask], end_yr)
            if r:
                parts.append(f"{r['coef']:>8.3f}{stars(r['p']):3s} ({r['se']:.3f}) N={r['n']}")
            else:
                parts.append(f"{'N/A':>30s}")
        print(f"  1988-{end_yr:<8} {parts[0]:>30s}   {parts[1]:>30s}")


def main():
    print("Loading panel ...")
    df = load_panel()
    print(f"  {len(df):,} obs, {df['paper'].nunique()} papers, "
          f"{df['cz'].nunique()} CZs")

    part1_circulation_response(df)
    part2_ad_by_initial_lean(df)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
