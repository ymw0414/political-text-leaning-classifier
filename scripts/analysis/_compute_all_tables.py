"""Compute all table numbers from fresh panel for verification."""
import sys, os, json
import numpy as np
import pandas as pd
import pyfixest as pf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
import pipeline_config as cfg

PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
NAFTA_YEAR = 1994
END_YEAR = 2004

STATE_TO_DIVISION = {
    9:1,23:1,25:1,33:1,44:1,50:1,
    34:2,36:2,42:2,
    17:3,18:3,26:3,39:3,55:3,
    19:4,20:4,27:4,29:4,31:4,38:4,46:4,
    10:5,11:5,12:5,13:5,24:5,37:5,45:5,51:5,54:5,
    1:6,21:6,28:6,47:6,
    5:7,22:7,40:7,48:7,
    4:8,8:8,16:8,30:8,32:8,35:8,49:8,56:8,
    2:9,6:9,15:9,41:9,53:9,
}

df = pd.read_parquet(PANEL_PATH)
df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
df = df[df["year"] <= END_YEAR].copy()
df["state_fips"] = (df["fips"] // 1000).astype(int)
df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
df["paper_id"] = df["paper"].astype("category").cat.codes
df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

years = sorted(df["year"].unique())
base_yr = years[0]
for yr in years:
    if yr == base_yr: continue
    df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)
    df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
manu_rhs = " + ".join(manu_vars)
china_rhs = " + ".join(china_vars)

pval_key = "Pr(>|t|)"

print(f"Panel: {len(df)} obs, {df['paper'].nunique()} papers, {df['cz'].nunique()} CZs")

# ============================================================
# 1. SHORT/LONG DECOMPOSITION
# ============================================================
print("\n=== SHORT/LONG DECOMPOSITION ===")
df["short"] = ((df["year"] >= 1994) & (df["year"] <= 1998)).astype(int)
df["long"] = (df["year"] >= 1999).astype(int)
df["vuln_x_short"] = df["vulnerability1990_scaled"] * df["short"]
df["vuln_x_long"] = df["vulnerability1990_scaled"] * df["long"]

shortlong_results = {}
for depvar in ["share_R", "share_D", "net_slant_norm", "right_norm", "left_norm"]:
    fml = f"{depvar} ~ vuln_x_short + vuln_x_long + {china_rhs} + {manu_rhs} | paper_id + year + division^year"
    m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
    t = m.tidy()
    s = t.loc["vuln_x_short"]
    l = t.loc["vuln_x_long"]
    print(f"  {depvar}: short={s['Estimate']:.4f} ({s['Std. Error']:.4f}), p={s[pval_key]:.4f}")
    print(f"  {depvar}: long ={l['Estimate']:.4f} ({l['Std. Error']:.4f}), p={l[pval_key]:.4f}")
    shortlong_results[depvar] = {
        "short_coef": float(s["Estimate"]), "short_se": float(s["Std. Error"]),
        "short_pval": float(s[pval_key]),
        "long_coef": float(l["Estimate"]), "long_se": float(l["Std. Error"]),
        "long_pval": float(l[pval_key]),
        "N": int(m._N)
    }

# Save updated shortlong
out = cfg.TAB_DIR
out.mkdir(parents=True, exist_ok=True)
with open(out / "did_shortlong.json", "w") as f:
    json.dump(shortlong_results, f, indent=2)
# Also save to main output
main_out = Path(os.environ["SHIFTING_SLANT_DIR"]) / "output" / "tables"
main_out.mkdir(parents=True, exist_ok=True)
with open(main_out / "did_shortlong.json", "w") as f:
    json.dump(shortlong_results, f, indent=2)
print(f"  Saved shortlong to both locations")

# ============================================================
# 2. EDUCATION HETEROGENEITY
# ============================================================
print("\n=== EDUCATION HETEROGENEITY ===")
edu_col = "bachelor_higher1990"
paper_edu = df.groupby("paper").first()[edu_col]
median_edu = paper_edu.median()
df["high_edu"] = (df[edu_col] > median_edu).astype(int)
df["vuln_x_post_x_highedu"] = df["vuln_x_post"] * df["high_edu"]

# Standardized continuous
edu_std = paper_edu.std()
df["edu_std"] = (df[edu_col] - paper_edu.mean()) / edu_std
df["vuln_x_post_x_edu"] = df["vuln_x_post"] * df["edu_std"]

# Col 1: Median split
fml1 = f"share_R ~ vuln_x_post + vuln_x_post_x_highedu + {china_rhs} + {manu_rhs} | paper_id + year + division^year"
m1 = pf.feols(fml1, data=df, vcov={"CRV1": "cz"})
t1 = m1.tidy()
base = t1.loc["vuln_x_post"]
inter = t1.loc["vuln_x_post_x_highedu"]
implied = base["Estimate"] + inter["Estimate"]
print(f"  Col 1 (median split):")
print(f"    Base:    {base['Estimate']:.4f} ({base['Std. Error']:.4f}), p={base[pval_key]:.4f}")
print(f"    Inter:   {inter['Estimate']:.4f} ({inter['Std. Error']:.4f}), p={inter[pval_key]:.4f}")
print(f"    Implied: {implied:.4f}")
print(f"    N={m1._N}")
n_low = df[df["high_edu"]==0]["paper"].nunique()
n_high = df[df["high_edu"]==1]["paper"].nunique()
print(f"    Low-edu papers: {n_low}, High-edu papers: {n_high}")
print(f"    Median education: {median_edu*100:.1f}%")
print(f"    SD education: {edu_std*100:.1f}%")

# Col 2: Continuous
fml2 = f"share_R ~ vuln_x_post + vuln_x_post_x_edu + {china_rhs} + {manu_rhs} | paper_id + year + division^year"
m2 = pf.feols(fml2, data=df, vcov={"CRV1": "cz"})
t2 = m2.tidy()
cont = t2.loc["vuln_x_post_x_edu"]
base2 = t2.loc["vuln_x_post"]
print(f"  Col 2 (continuous):")
print(f"    Base:    {base2['Estimate']:.4f} ({base2['Std. Error']:.4f})")
print(f"    Inter:   {cont['Estimate']:.4f} ({cont['Std. Error']:.4f}), p={cont[pval_key]:.4f}")

# ============================================================
# 3. SLANT HETEROGENEITY
# ============================================================
print("\n=== SLANT HETEROGENEITY ===")
pre = df[df["year"] < NAFTA_YEAR].groupby("paper")["share_R"].mean()
median_slant = pre.median()
df = df.merge(pre.rename("pre_share_R"), on="paper", how="left")
df["high_slant"] = (df["pre_share_R"] > median_slant).astype(int)
df["vuln_x_post_x_highslant"] = df["vuln_x_post"] * df["high_slant"]

# Standardized continuous
slant_std = pre.std()
df["slant_std"] = (df["pre_share_R"] - pre.mean()) / slant_std
df["vuln_x_post_x_slant"] = df["vuln_x_post"] * df["slant_std"]

# Col 1: Median split
fml3 = f"share_R ~ vuln_x_post + vuln_x_post_x_highslant + {china_rhs} + {manu_rhs} | paper_id + year + division^year"
m3 = pf.feols(fml3, data=df, vcov={"CRV1": "cz"})
t3 = m3.tidy()
base_s = t3.loc["vuln_x_post"]
inter_s = t3.loc["vuln_x_post_x_highslant"]
implied_s = base_s["Estimate"] + inter_s["Estimate"]
print(f"  Col 1 (median split):")
print(f"    Base:    {base_s['Estimate']:.4f} ({base_s['Std. Error']:.4f}), p={base_s[pval_key]:.4f}")
print(f"    Inter:   {inter_s['Estimate']:.4f} ({inter_s['Std. Error']:.4f}), p={inter_s[pval_key]:.4f}")
print(f"    Implied: {implied_s:.4f}")
n_dleaning = df[df["high_slant"]==0]["paper"].nunique()
n_rleaning = df[df["high_slant"]==1]["paper"].nunique()
print(f"    D-leaning papers: {n_dleaning}, R-leaning papers: {n_rleaning}")
print(f"    Median pre-NAFTA share_R: {median_slant:.4f}")
print(f"    SD pre-NAFTA share_R: {slant_std:.4f}")

# Col 2: Continuous
fml4 = f"share_R ~ vuln_x_post + vuln_x_post_x_slant + {china_rhs} + {manu_rhs} | paper_id + year + division^year"
m4 = pf.feols(fml4, data=df, vcov={"CRV1": "cz"})
t4 = m4.tidy()
cont_s = t4.loc["vuln_x_post_x_slant"]
base_s2 = t4.loc["vuln_x_post"]
print(f"  Col 2 (continuous):")
print(f"    Base:    {base_s2['Estimate']:.4f} ({base_s2['Std. Error']:.4f})")
print(f"    Inter:   {cont_s['Estimate']:.4f} ({cont_s['Std. Error']:.4f}), p={cont_s[pval_key]:.4f}")

# ============================================================
# 4. VARIANCE DECOMPOSITION
# ============================================================
print("\n=== VARIANCE DECOMPOSITION ===")
total_var = df["share_R"].var()
m_paper = pf.feols("share_R ~ 1 | paper_id", data=df)
r2_paper = m_paper._r2
m_year = pf.feols("share_R ~ 1 | year", data=df)
r2_year = m_year._r2
m_both = pf.feols("share_R ~ 1 | paper_id + year", data=df)
r2_both = m_both._r2
residual = 1 - r2_both
print(f"  Total variance: {total_var:.6f}")
print(f"  Paper FE R2: {r2_paper:.4f} ({r2_paper*100:.1f}%)")
print(f"  Year FE R2: {r2_year:.4f} ({r2_year*100:.1f}%)")
print(f"  Both R2: {r2_both:.4f} ({r2_both*100:.1f}%)")
print(f"  Residual: {residual:.4f} ({residual*100:.1f}%)")

# Autocorrelation
resid = m_both.resid()
resid_df = df[["paper", "year"]].copy()
resid_df["resid"] = resid
resid_df = resid_df.sort_values(["paper", "year"])
resid_df["resid_lag"] = resid_df.groupby("paper")["resid"].shift(1)
valid = resid_df.dropna(subset=["resid_lag"])
rho = valid["resid"].corr(valid["resid_lag"])
print(f"  Autocorrelation (rho): {rho:.4f}")

# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================
print("\n=== SUMMARY STATISTICS ===")
for col in ["share_R", "vulnerability1990_scaled", "china_shock", "manushare1990",
            "bachelor_higher1990"]:
    s = df[col]
    print(f"  {col}: N={s.notna().sum()}, mean={s.mean():.3f}, sd={s.std():.3f}, "
          f"p25={s.quantile(0.25):.3f}, p50={s.quantile(0.5):.3f}, p75={s.quantile(0.75):.3f}")

if "income_pc1989" in df.columns:
    inc = df["income_pc1989"] / 1000
    print(f"  income_pc1989 ($1000s): mean={inc.mean():.1f}, sd={inc.std():.1f}, "
          f"p25={inc.quantile(0.25):.1f}, p50={inc.quantile(0.5):.1f}, p75={inc.quantile(0.75):.1f}")

arts = df["n_articles"] / 1000
print(f"  n_articles (1000s): mean={arts.mean():.1f}, sd={arts.std():.1f}, "
      f"p25={arts.quantile(0.25):.1f}, p50={arts.quantile(0.5):.1f}, p75={arts.quantile(0.75):.1f}")

# ============================================================
# 6. PANEL DESCRIPTION
# ============================================================
print("\n=== PANEL DESCRIPTION ===")
yearly = df.groupby("year").agg(
    papers=("paper", "nunique"),
    czs=("cz", "nunique"),
    articles=("n_articles", "sum"),
    arts_per_paper=("n_articles", "mean")
)
for yr, row in yearly.iterrows():
    print(f"  {yr}: {row['papers']:>3} papers, {row['czs']:>3} CZs, "
          f"{row['articles']:>10,.0f} articles, {row['arts_per_paper']:>8,.0f} per paper")
print(f"  Mean: {yearly['papers'].mean():.0f} papers, {yearly['czs'].mean():.0f} CZs, "
      f"{yearly['articles'].mean():,.0f} articles, {yearly['arts_per_paper'].mean():,.0f} per paper")
print(f"  Total articles: {df['n_articles'].sum():,}")

print("\nDone.")
