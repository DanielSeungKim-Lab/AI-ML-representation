"""Main analysis: cohort counts, Tables 1 and 2 and the supplementary analyses
against the matched comparator trials. Writes RESULTS-01 to RESULTS-14; RESULTS-01
lists every value. M1_COHORT=category1 repeats the analysis for the 53 category-1
trials (DATA-08); outputs then carry the suffix -category1.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(HERE, f)

AI = pd.read_csv(P("DATA-01-ai-cohort.csv"))
POSTED = pd.read_csv(P("DATA-02-ai-posted-results.csv"))
ETH = pd.read_csv(P("DATA-03-ai-ethnicity.csv"))
CMP_MATCHED = pd.read_csv(P("DATA-04-comparator-counts.csv"))
# Comparator trials whose race counts do not reconcile with enrollment are excluded.
CMP = CMP_MATCHED[CMP_MATCHED.race_reconciles].copy()
MATCH = pd.read_csv(P("DATA-05-comparator-matching.csv"))
SHARE = pd.read_csv(P("DATA-06-census-adult-shares.csv")).set_index("group")["pop_share_adult"]

COHORT = os.environ.get("M1_COHORT", "all107")
assert COHORT in ("category1", "all107"), COHORT
if COHORT == "category1":
    ROLE = pd.read_csv(P("DATA-08-ai-role-classification.csv"))
    KEEP = set(ROLE[ROLE.final_category == 1].nct_id)
    AI = AI[AI.nct_id.isin(KEEP)].reset_index(drop=True)
    ETH = ETH[ETH.nct_id.isin(KEEP)].reset_index(drop=True)
SUFFIX = "" if COHORT == "all107" else "-category1"
PO = lambda f: P(f.replace(".csv", SUFFIX + ".csv"))

# DATA-05 covers all 530 drawn trials; enrollment totals use the retained trials.
MATCH_COHORT = MATCH[MATCH.nct_id.isin(set(CMP.nct_id))]

KNOWN = ["White", "Black", "Asian", "AIAN", "NHPI", "Multiple", "Other"]
RACE = ["Black", "White", "Asian", "AIAN", "NHPI"]
B = 4000
SEED = 7


def ppr_ai(group):
    """Per-trial PPR in the AI/ML cohort, from the stored category shares."""
    return (AI[group + "_sh"] / SHARE[group]).values


def ppr_cmp(group):
    """Per-trial PPR among comparator trials reporting at least one known race."""
    d = CMP[CMP[KNOWN].sum(axis=1) > 0]
    return ((d[group] / d[KNOWN].sum(axis=1)) / SHARE[group]).values


def ppr_eth_ai():
    return ((ETH.hisp / (ETH.hisp + ETH.nothisp)) / SHARE["Hispanic"]).values


def ppr_eth_cmp():
    d = CMP[(CMP.hisp + CMP.nothisp) > 0]
    return ((d.hisp / (d.hisp + d.nothisp)) / SHARE["Hispanic"]).values


def boot_ci(v):
    """Percentile bootstrap interval for the median, resampling trials.

    A fresh generator is seeded per call so the interval does not depend on the
    order in which the analyses are run.
    """
    v = np.asarray(v, dtype=float)
    draws = np.random.default_rng(SEED).choice(v, (B, len(v)), replace=True)
    s = np.median(draws, axis=1)
    return float(np.median(v)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ---------------------------------------------------------------- cohort counts
rec = {}
rec["AI/ML trials in primary cohort"] = len(AI)
rec["AI/ML trials with posted baseline results"] = len(POSTED)
rec["AI/ML participants"] = AI.baseline_denominator.sum()
rec["AI/ML participants in a known race category"] = AI.known_sum.sum()
rec["AI/ML participants with unknown race"] = AI.Unknown.sum()
rec["AI/ML trials reporting ethnicity"] = len(ETH)
rec["AI/ML trials reporting ethnicity, %"] = 100 * len(ETH) / len(AI)
rec["Comparator trials matched"] = len(CMP_MATCHED)
rec["Comparator trials excluded by reconciliation"] = int((~CMP_MATCHED.race_reconciles).sum())
rec["Comparator trials"] = len(CMP)
rec["Comparator participants"] = MATCH_COHORT.enroll.sum()
rec["Comparator trials reporting race"] = int((CMP[KNOWN].sum(axis=1) > 0).sum())
rec["Comparator trials reporting race, %"] = 100 * (CMP[KNOWN].sum(axis=1) > 0).mean()
rec["Comparator trials reporting ethnicity"] = int(((CMP.hisp + CMP.nothisp) > 0).sum())
rec["Comparator trials reporting ethnicity, %"] = 100 * ((CMP.hisp + CMP.nothisp) > 0).mean()
rec["Comparator trials reporting race only as unknown"] = int(
    ((CMP[KNOWN].sum(axis=1) == 0) & (CMP.Unknown > 0)).sum())
rec["Median enrollment, AI/ML"] = AI.baseline_denominator.median()
rec["Median enrollment, comparator"] = MATCH_COHORT.enroll.median()
outside = AI.race_count_total - AI.known_sum - AI.Unknown
rec["AI/ML participants in the analyzed race categories"] = (AI.known_sum + AI.Unknown).sum()
rec["AI/ML participants in the posted race measures"] = AI.race_count_total.sum()
rec["AI/ML participants in the race measure outside the race categories"] = outside.sum()
rec["AI/ML trials with participants outside the race categories"] = int((outside > 0).sum())
rec["AI/ML trials with combined race and ethnicity categories"] = int(AI.combined_race_ethnicity.sum())
rec["Comparator trials with combined race and ethnicity categories"] = int(CMP[CMP[KNOWN].sum(axis=1) > 0].combined_race_ethnicity.sum())
rec["AI/ML enrollment not covered by the race measure"] = (AI.baseline_denominator - AI.race_count_total).sum()
pk = POSTED[KNOWN].sum(axis=1)
prec = (POSTED.race_count_total / POSTED.baseline_denominator).between(0.95, 1.05)
rec["Posted AI/ML trials with no known race category"] = int((pk == 0).sum())
rec["Posted AI/ML trials reporting race"] = int((pk > 0).sum())
rec["Posted AI/ML trials with race not reconciling"] = int(((pk > 0) & ~prec).sum())
rec["AI/ML trials with a comparator stratum"] = int((AI.start_year >= 2015).sum())
rec["Matched comparator trials reporting race"] = int((CMP_MATCHED[KNOWN].sum(axis=1) > 0).sum())
c_race, c_eth = CMP[KNOWN].sum(axis=1) > 0, (CMP.hisp + CMP.nothisp) > 0
rec["Comparator trials reporting race and ethnicity"] = int((c_race & c_eth).sum())
rec["Comparator trials reporting neither"] = int((~c_race & ~c_eth).sum())

# ------------------------------------------------------------------- Table 1
# AIAN and NHPI trial medians are 0 by construction; a parity-calibrated reference is given.
def parity_calibration(R=2000):
    """Trial-median PPR expected under exactly population-proportional enrollment
    at the observed trial sizes (Supplementary Methods). Monte Carlo quantities:
    rerunning the submitted analysis reproduces the published calibrated medians
    to within 0.002 and their limits to within 0.006."""
    rng = np.random.default_rng(SEED)
    p = np.array([SHARE[g] for g in RACE5] + [1 - sum(SHARE[g] for g in RACE5)])
    m = AI.known_sum.values.astype(int)
    X = np.stack([rng.multinomial(m, p) for _ in range(R)])
    out = {}
    for k, g in enumerate(RACE5):
        M = np.median(X[:, :, k] / m / SHARE[g], axis=1)
        out[g] = (float(np.median(M)), float(np.percentile(M, 2.5)), float(np.percentile(M, 97.5)))
    e = (ETH.hisp + ETH.nothisp).values.astype(int)
    Y = np.stack([rng.binomial(e, SHARE["Hispanic"]) for _ in range(R)])
    M = np.median(Y / e / SHARE["Hispanic"], axis=1)
    out["Hispanic"] = (float(np.median(M)), float(np.percentile(M, 2.5)), float(np.percentile(M, 97.5)))
    return out


RACE5 = ["White", "Black", "Asian", "AIAN", "NHPI"]
PARITY = parity_calibration()
NOT_ESTIMABLE = {g: round(PARITY[g][0], 2) for g in ("AIAN", "NHPI")}
pd.DataFrame([{"group": g, "trials": len(ETH) if g == "Hispanic" else len(AI), "pop_share": SHARE[g],
               "observed_trial_median": float(np.median(ppr_eth_ai() if g == "Hispanic" else ppr_ai(g))),
               "parity_calibrated_median": v[0], "calibrated_2_5": v[1], "calibrated_97_5": v[2]}
              for g, v in PARITY.items()]).to_csv(PO("RESULTS-14-parity-calibration.csv"), index=False)

t1 = []
for g in RACE + ["Hispanic"]:
    a = ppr_eth_ai() if g == "Hispanic" else ppr_ai(g)
    c = ppr_eth_cmp() if g == "Hispanic" else ppr_cmp(g)
    am, alo, ahi = boot_ci(a)
    cm, clo, chi = boot_ci(c)
    p = mannwhitneyu(a, c).pvalue
    ne = g in NOT_ESTIMABLE
    t1.append({"group": g, "pop_share": SHARE[g], "ai_n": len(a),
               "ai_median": None if ne else am, "ai_lo": None if ne else alo,
               "ai_hi": None if ne else ahi, "cmp_n": len(c),
               "cmp_median": None if ne else cm, "cmp_lo": None if ne else clo,
               "cmp_hi": None if ne else chi, "p_value": p,
               "estimable": "no" if ne else "yes",
               "parity_calibrated_median": NOT_ESTIMABLE.get(g),
               "note": ("not estimable at this estimand; most trials enrolled no "
                        "participant in this category") if ne else ""})
    if ne:
        continue
    rec[f"Table 1 {g} AI/ML median"] = am
    rec[f"Table 1 {g} AI/ML CI lower"] = alo
    rec[f"Table 1 {g} AI/ML CI upper"] = ahi
    rec[f"Table 1 {g} comparator median"] = cm
    rec[f"Table 1 {g} comparator CI lower"] = clo
    rec[f"Table 1 {g} comparator CI upper"] = chi
    rec[f"Table 1 {g} P value"] = p
T1 = pd.DataFrame(t1)
T1.to_csv(PO("RESULTS-02-table1-ppr.csv"), index=False)
both = CMP[c_race & c_eth]
v_both = ((both.hisp / (both.hisp + both.nothisp)) / SHARE["Hispanic"]).values
rec["Table 1 Hispanic comparator median, trials also reporting race"] = float(np.median(v_both))
rec["Table 1 Hispanic P value, comparator trials also reporting race"] = mannwhitneyu(ppr_eth_ai(), v_both).pvalue

# -------------------------------------------------- trials below 0.8 of parity
for g in ["White", "Black", "Asian"]:
    rec[f"Trials with {g} PPR below 0.8"] = int((ppr_ai(g) < 0.8).sum())
rec["Trials with Hispanic PPR below 0.8"] = int((ppr_eth_ai() < 0.8).sum())

# -------------------------------------------------------------- period medians
PERIODS = [("2015-2019", 2015, 2019), ("2020-2021", 2020, 2021), ("2022-2025", 2022, 2025)]
for lab, lo, hi in PERIODS:
    m = AI.start_year.between(lo, hi)
    rec[f"Period {lab} trials"] = int(m.sum())
    for g in ["Black", "White", "Asian"]:
        rec[f"Period {lab} {g} median"] = float(np.median((AI.loc[m, g + "_sh"] / SHARE[g]).values))
    e = ETH[ETH.nct_id.isin(AI.loc[m, "nct_id"])]
    rec[f"Period {lab} trials reporting ethnicity"] = len(e)
    rec[f"Period {lab} Hispanic median"] = float(
        np.median(((e.hisp / (e.hisp + e.nothisp)) / SHARE["Hispanic"]).values))

# ------------------------------------------------------------- leave-one-out
for g in RACE + ["Hispanic"]:
    a, c = (ppr_eth_ai(), ppr_eth_cmp()) if g == "Hispanic" else (ppr_ai(g), ppr_cmp(g))
    p0 = mannwhitneyu(a, c).pvalue
    loo_p = [mannwhitneyu(np.delete(a, i), c).pvalue for i in range(len(a))]
    loo_m = [float(np.median(np.delete(a, i))) for i in range(len(a))]
    rec[f"Leave-one-out {g} P value, minimum"] = min(loo_p)
    rec[f"Leave-one-out {g} P value, maximum"] = max(loo_p)
    rec[f"Leave-one-out {g} removals crossing P = 0.05"] = sum((p < 0.05) != (p0 < 0.05) for p in loo_p)
    rec[f"Leave-one-out {g} median, minimum"] = min(loo_m)
    rec[f"Leave-one-out {g} median, maximum"] = max(loo_m)

# ------------------------------------------------- label-mapping sensitivity
CR_MAP = CMP[CMP[KNOWN].sum(axis=1) > 0]
for g in ["Black", "White", "Asian"]:
    for lab, ka, kc in [("excluding combined race and ethnicity categories", ~AI.combined_race_ethnicity, ~CR_MAP.combined_race_ethnicity),
                        ("excluding Asian or Pacific Islander categories", ~AI.asian_pacific_combined, ~CR_MAP.asian_pacific_combined)]:
        a = (AI.loc[ka, g + "_sh"] / SHARE[g]).values
        c = ((CR_MAP.loc[kc, g] / CR_MAP.loc[kc, KNOWN].sum(axis=1)) / SHARE[g]).values
        rec[f"Mapping sensitivity {lab}, {g} AI/ML trials"] = len(a)
        rec[f"Mapping sensitivity {lab}, {g} AI/ML median"] = float(np.median(a))
        rec[f"Mapping sensitivity {lab}, {g} comparator trials"] = len(c)
        rec[f"Mapping sensitivity {lab}, {g} comparator median"] = float(np.median(c))
        rec[f"Mapping sensitivity {lab}, {g} P value"] = mannwhitneyu(a, c).pvalue

# ------------------------------------------------------------------- sponsor
for cls, lab in [("FED", "federal"), ("INDUSTRY", "industry"), ("OTHER", "other")]:
    v = (AI.loc[AI.sponsor_class == cls, "Black_sh"] / SHARE["Black"]).values
    m, lo, hi = boot_ci(v)
    rec[f"Sponsor {lab} Black median"] = m
    rec[f"Sponsor {lab} Black CI lower"] = lo
    rec[f"Sponsor {lab} Black CI upper"] = hi
    rec[f"Sponsor {lab} trials"] = len(v)
rec["Sponsor Kruskal-Wallis P value"] = kruskal(
    *[(AI.loc[AI.sponsor_class == c2, "Black_sh"] / SHARE["Black"]).values
      for c2 in ["FED", "INDUSTRY", "OTHER"]]).pvalue

# ------------------------------------------------------------------ geography
usable = POSTED[POSTED.usable_race_ethnicity]
rec["Trials with usable race counts"] = len(usable)
rec["Usable trials with all sites in the US"] = int(
    ((usable.us_sites == usable.n_sites) & (usable.n_sites > 0)).sum())
rec["Usable trials with mixed US and non-US sites"] = int(
    ((usable.us_sites > 0) & (usable.us_sites < usable.n_sites)).sum())
rec["Usable trials with no US site"] = int(((usable.us_sites == 0) & (usable.n_sites > 0)).sum())
rec["Usable trials with no location records"] = int((usable.n_sites == 0).sum())
no_us = POSTED[(POSTED.us_sites == 0) & (POSTED.n_sites > 0)]
all_us = POSTED[(POSTED.us_sites == POSTED.n_sites) & (POSTED.n_sites > 0)]
nu_ok = int(no_us.usable_race_ethnicity.sum())
au_ok = int(all_us.usable_race_ethnicity.sum())
rec["No-US-site trials with usable race counts"] = nu_ok
rec["No-US-site trials"] = len(no_us)
rec["US-only trials with usable race counts"] = au_ok
rec["US-only trials"] = len(all_us)
rec["Geography Fisher exact P value"] = fisher_exact(
    [[nu_ok, len(no_us) - nu_ok], [au_ok, len(all_us) - au_ok]])[1]
any_us = POSTED[(POSTED.us_sites > 0) & POSTED.usable_race_ethnicity]
rec["Any-US-site cohort trials"] = len(any_us)
rec["Any-US-site cohort Black median"] = float(np.median(
    ((any_us.Black / any_us[KNOWN].sum(axis=1)) / SHARE["Black"]).values))
rec["No-US-site trials without usable race counts"] = len(no_us) - nu_ok
rec["No-US-site trials listing all five OMB race categories"] = int((no_us.omb_race_categories_listed == 5).sum())

# ------------------------------------------------------------ pooled estimates
for g in RACE:
    rec[f"Pooled {g} PPR"] = float((AI[g].sum() / AI.known_sum.sum()) / SHARE[g])

# ------------------------------------- ethnicity denominators that are degenerate
deg_ai = ETH[ETH.nothisp == 0]
cmp_eth = CMP[(CMP.hisp + CMP.nothisp) > 0]
deg_cmp = cmp_eth[cmp_eth.nothisp == 0]
rec["AI/ML trials with a Hispanic count and no non-Hispanic count"] = len(deg_ai)
rec["Comparator trials with a Hispanic count and no non-Hispanic count"] = len(deg_cmp)
rec["AI/ML trials with ethnicity from a combined race and ethnicity list"] = int(ETH.ethnicity_from_combined_list.sum())
rec["Comparator trials with ethnicity from a combined race and ethnicity list"] = int(cmp_eth.ethnicity_from_combined_list.sum())
ka = ETH[ETH.nothisp > 0]
kc = cmp_eth[cmp_eth.nothisp > 0]
va = ((ka.hisp / (ka.hisp + ka.nothisp)) / SHARE["Hispanic"]).values
vc = ((kc.hisp / (kc.hisp + kc.nothisp)) / SHARE["Hispanic"]).values
m, lo, hi = boot_ci(va)
rec["Non-degenerate AI/ML Hispanic median"] = m
rec["Non-degenerate AI/ML Hispanic CI lower"] = lo
rec["Non-degenerate AI/ML Hispanic CI upper"] = hi
m, lo, hi = boot_ci(vc)
rec["Non-degenerate comparator Hispanic median"] = m
rec["Non-degenerate comparator Hispanic CI lower"] = lo
rec["Non-degenerate comparator Hispanic CI upper"] = hi
rec["Non-degenerate Hispanic P value"] = mannwhitneyu(va, vc).pvalue
# Trials enrolling only Hispanic/Latinx participants by design (registry eligibility criteria).
BY_DESIGN = {"NCT06472297", "NCT03839212", "NCT04430335", "NCT05292248", "NCT05391438"}
deg = pd.concat([deg_ai.merge(AI[["nct_id", "baseline_denominator"]], on="nct_id"),
                 deg_cmp[["nct_id", "hisp", "baseline_denominator"]]])
rest = deg[~deg.nct_id.isin(BY_DESIGN)]
rec["Hispanic-count-only trials enrolling only Hispanic/Latinx participants by design"] = int(deg.nct_id.isin(BY_DESIGN).sum())
rec["Other Hispanic-count-only trials, median Hispanic share of enrollment, %"] = 100 * float(
    (rest.hisp / rest.baseline_denominator).median())

# ------------------------------------------------------- frequency-match balance
# 106 AI/ML trials (the 2008 trial has no comparator stratum) and 530 drawn trials.
MBANDS = [0, 100, 300, 1000, 5000, np.inf]
MBAND_LABELS = ["1-100", "101-300", "301-1,000", "1,001-5,000", ">5,000"]


def match_period(y):
    return ("2015-2017" if y < 2018 else "2018-2019" if y < 2020 else
            "2020-2021" if y < 2022 else "2022-2025")


_ai = AI[AI.start_year >= 2015].assign(
    period=lambda d: d.start_year.map(match_period),
    band=lambda d: pd.cut(d.baseline_denominator, MBANDS, labels=MBAND_LABELS, right=True))  # bands are right-closed
_cm = MATCH.assign(period=lambda d: d.year.map(match_period),
                   band=lambda d: d.sz.map(dict(enumerate(MBAND_LABELS))))
bal = pd.concat([_ai.groupby(["period", "band"], observed=True).size().rename("ai_trials"),
                 _cm.groupby(["period", "band"], observed=True).size().rename("comparator_trials")],
                axis=1).fillna(0).astype(int).reset_index()
bal = bal[bal.ai_trials > 0].copy()
bal["comparator_to_ai_ratio"] = (bal.comparator_trials / bal.ai_trials).round(2)
bal.to_csv(PO("RESULTS-05-matching-balance.csv"), index=False)

# ---------------------------------------------------------------- per-trial PPRs
per = AI[["nct_id", "start_year", "baseline_denominator", "sponsor_class"]].copy()
for g in RACE:
    per[g + "_ppr"] = ppr_ai(g)
per = per.merge(ETH.assign(Hispanic_ppr=ppr_eth_ai())[["nct_id", "Hispanic_ppr"]],
                on="nct_id", how="left")
per.sort_values("Black_ppr", ascending=False).to_csv(PO("RESULTS-04-per-trial-ppr.csv"), index=False)

rec["Table 1 AIAN P value"] = mannwhitneyu(ppr_ai("AIAN"), ppr_cmp("AIAN")).pvalue
rec["Table 1 NHPI P value"] = mannwhitneyu(ppr_ai("NHPI"), ppr_cmp("NHPI")).pvalue

# --------------------------------------------------------------- supplements

BANDS = [0, 100, 300, 1000, 5000, np.inf]
BAND_LABELS = ["1-100", "101-300", "301-1,000", "1,001-5,000", ">5,000"]
SPONSOR = {"INDUSTRY": "Industry", "FED": "US federal, non-NIH",
           "NIH": "National Institutes of Health",
           "OTHER": "Other (academic, hospital, foundation)",
           "NETWORK": "Research network", "OTHER_GOV": "Other government"}


def period(y):
    if y < 2015:
        return "Before 2015"
    return "2015-2017" if y < 2018 else ("2018-2019" if y < 2020 else
           ("2020-2021" if y < 2022 else "2022-2025"))


# Enrollment bands are right-closed; the comparator band is the one recorded in DATA-05.
ai_r = AI.copy()
ai_r["period"] = ai_r.start_year.apply(period)
ai_r["band"] = pd.cut(ai_r.baseline_denominator, BANDS, labels=BAND_LABELS, right=True)
ai_r["sponsor_label"] = ai_r.sponsor_class.map(SPONSOR).fillna(ai_r.sponsor_class)

cmp_r = CMP[CMP[KNOWN].sum(axis=1) > 0].merge(MATCH[["nct_id", "sz"]], on="nct_id", how="left")
cmp_r["period"] = cmp_r.year.apply(period)
cmp_r["band"] = cmp_r.sz.map(dict(enumerate(BAND_LABELS)))
cmp_r["sponsor_label"] = cmp_r.sponsor.map(SPONSOR).fillna(cmp_r.sponsor)

bal = []
for cov, col, levels, matched_on in [
        ("Trial start period", "period",
         ["Before 2015", "2015-2017", "2018-2019", "2020-2021", "2022-2025"], "Yes"),
        ("Enrollment band", "band", BAND_LABELS, "Yes"),
        ("Sponsor class", "sponsor_label", list(SPONSOR.values()), "No")]:
    for lv in levels:
        n1, n2 = int((ai_r[col] == lv).sum()), int((cmp_r[col] == lv).sum())
        p1, p2 = n1 / len(ai_r), n2 / len(cmp_r)
        v = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
        bal.append({"covariate": cov, "level": lv,
                    "AI_no_pct": f"{n1} ({100 * p1:.1f})",
                    "comparator_no_pct": f"{n2} ({100 * p2:.1f})",
                    "standardized_difference": f"{(p1 - p2) / np.sqrt(v):+.3f}" if v > 0 else "+0.000",
                    "frequency_matched_on": matched_on})
pd.DataFrame(bal).to_csv(PO("RESULTS-07-covariate-balance.csv"), index=False)

# Characteristics of the analyzed cohorts (all retained comparator trials).
cmp_all = CMP.merge(MATCH[["nct_id", "sz"]], on="nct_id", how="left")
for lab, per, band, spon in [("AI/ML", ai_r.period, ai_r.band, ai_r.sponsor_label),
                             ("comparator", cmp_all.year.apply(period), cmp_all.sz.map(dict(enumerate(BAND_LABELS))),
                              cmp_all.sponsor.map(SPONSOR).fillna(cmp_all.sponsor))]:
    for lv, n in pd.concat([per.value_counts(), band.value_counts(), spon.value_counts()]).items():
        rec[f"Characteristics, {lab}: {lv}"] = int(n)


def weighted_median(v, w):
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    return float(v[np.searchsorted(np.cumsum(w) / np.sum(w), 0.5)])


def van_elteren(x_all, y_all, sx, sy):
    """Stratified Wilcoxon rank-sum with the 1/(n+1) stratum weight."""
    from scipy.stats import rankdata, norm
    T = E = V = 0.0
    for h in sorted(set(sx) | set(sy)):
        x, y = np.asarray(x_all)[np.asarray(sx) == h], np.asarray(y_all)[np.asarray(sy) == h]
        n1, n2, n = len(x), len(y), len(x) + len(y)
        if n1 == 0 or n2 == 0 or n < 2:
            continue
        r = rankdata(np.concatenate([x, y]))
        w = 1.0 / (n + 1)
        T += w * r[:n1].sum()
        E += w * n1 * (n + 1) / 2
        V += w ** 2 * n1 * n2 / (n * (n - 1)) * (np.sum(r ** 2) - n * ((n + 1) / 2) ** 2)
    return float(2 * (1 - norm.cdf(abs((T - E) / np.sqrt(V)))))


# Sponsor classes are collapsed to the three that both arms populate.
def p_display(p):
    """Manuscript convention: <0.001; three decimals below 0.01; otherwise two."""
    return "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


coarse = {"INDUSTRY": "INDUSTRY", "FED": "FED"}
sa = AI.sponsor_class.map(coarse).fillna("OTHER").values
sc = cmp_r.sponsor.map(coarse).fillna("OTHER").values
va = ppr_ai("Black")
vc = (cmp_r.Black / cmp_r[KNOWN].sum(axis=1) / SHARE["Black"]).values
tgt, cur = pd.Series(sa).value_counts(normalize=True), pd.Series(sc).value_counts(normalize=True)
wt = np.array([tgt.get(s, 0) / cur.get(s, 1) for s in sc])
pd.DataFrame([
    {"analysis": ("Primary" if COHORT == "all107" else "Category 1") + ": frequency-matched on start year and enrollment band",
     "AI_median_PPR": f"{np.median(va):.2f}", "comparator_median_PPR": f"{np.median(vc):.2f}",
     "P_value": p_display(mannwhitneyu(va, vc).pvalue)},
    {"analysis": "Comparator standardized to the AI arm's sponsor-class mix",
     "AI_median_PPR": f"{np.median(va):.2f}",
     "comparator_median_PPR": f"{weighted_median(vc, wt):.2f}", "P_value": "-"},
    {"analysis": "Sponsor-stratified rank test (van Elteren) across both arms",
     "AI_median_PPR": f"{np.median(va):.2f}", "comparator_median_PPR": "-",
     "P_value": p_display(van_elteren(va, vc, sa, sc))},
]).to_csv(PO("RESULTS-08-sponsor-adjusted.csv"), index=False)
rec["Sponsor-stratified van Elteren P value"] = van_elteren(va, vc, sa, sc)
rec["Sponsor-standardized comparator Black median"] = weighted_median(vc, wt)

ai_eth, cmp_eth2 = ETH, CMP[(CMP.hisp + CMP.nothisp) > 0]
ai_nd, cmp_nd = ai_eth[ai_eth.nothisp > 0], cmp_eth2[cmp_eth2.nothisp > 0]


def fmt(v):
    m, lo, hi = boot_ci(v)
    return f"{m:.2f} ({lo:.2f} to {hi:.2f})"


pd.DataFrame([
    {"Analysis": "AI/ML trials, as published", "Trials": len(ai_eth),
     "Median Hispanic PPR (95% CI)": fmt(ppr_eth_ai())},
    {"Analysis": f"AI/ML trials, excluding {len(ai_eth) - len(ai_nd)} with no non-Hispanic count",
     "Trials": len(ai_nd),
     "Median Hispanic PPR (95% CI)": fmt((ai_nd.hisp / (ai_nd.hisp + ai_nd.nothisp) / SHARE["Hispanic"]).values)},
    {"Analysis": "Comparator trials, as published", "Trials": len(cmp_eth2),
     "Median Hispanic PPR (95% CI)": fmt(ppr_eth_cmp())},
    {"Analysis": f"Comparator trials, excluding {len(cmp_eth2) - len(cmp_nd)} with no non-Hispanic count",
     "Trials": len(cmp_nd),
     "Median Hispanic PPR (95% CI)": fmt((cmp_nd.hisp / (cmp_nd.hisp + cmp_nd.nothisp) / SHARE["Hispanic"]).values)},
]).to_csv(PO("RESULTS-11-degenerate-ethnicity.csv"), index=False)


# ------------------------------------------------- denominator sensitivity
# Adult five-category vs all-ages benchmark; known vs unknown-inclusive denominator.
SHARE_ALLAGES = {"White": 0.60508, "Black": 0.12128, "Asian": 0.05987,
                 "AIAN": 0.00998, "NHPI": 0.00198, "Hispanic": 0.19450}
den_known = AI.known_sum
den_unknown_included = AI.known_sum + AI.Unknown
dsrows = []
for g in RACE + ["Hispanic"]:
    if g == "Hispanic":
        kn = ETH.hisp / (ETH.hisp + ETH.nothisp)
        ui = ETH.hisp / (ETH.hisp + ETH.nothisp + ETH.eth_unk)
    else:
        kn, ui = AI[g] / den_known, AI[g] / den_unknown_included
    est = g not in ("AIAN", "NHPI")
    dsrows.append({
        "group": g, "pop_share_adult": SHARE[g], "pop_share_allages": SHARE_ALLAGES[g],
        "median_PPR_adult_known": round(float(np.median(kn / SHARE[g])), 4) if est else None,
        "median_PPR_allages_known": round(float(np.median(kn / SHARE_ALLAGES[g])), 4) if est else None,
        "median_PPR_adult_unknown_included": round(float(np.median(ui / SHARE[g])), 4) if est else None,
        "estimable": "yes" if est else "no",
        "note": "" if est else "not estimable at this estimand; the trial median is zero by construction"})
pd.DataFrame(dsrows).to_csv(PO("RESULTS-06-denominator-sensitivity.csv"), index=False)


# ------------------------------------------- largest-trial sensitivity
# Descriptive: removes the 3 trials with the most participants in the analyzed race categories.
def pooled_jackknife(num, den, share):
    """Pooled PPR with a delete-one-trial jackknife interval on the log scale."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    theta = (num.sum() / den.sum()) / share
    n = len(num)
    li = np.array([np.log(((num.sum() - num[i]) / (den.sum() - den[i])) / share)
                   if num.sum() - num[i] > 0 else np.nan for i in range(n)])
    se = np.sqrt((n - 1) / n * np.nansum((li - np.nanmean(li)) ** 2))
    return theta, float(np.exp(np.log(theta) - 1.96 * se)), float(np.exp(np.log(theta) + 1.96 * se))


fmt3 = lambda t: f"{t[0]:.2f} ({t[1]:.2f}-{t[2]:.2f})"
AI_cov = AI.assign(cover=AI.known_sum + AI.Unknown)
top3 = AI_cov.nlargest(3, "cover")
ai_rest, eth_rest = AI[~AI.nct_id.isin(top3.nct_id)], ETH[~ETH.nct_id.isin(top3.nct_id)]
ltrows = []
for g in RACE + ["Hispanic"]:
    if g == "Hispanic":
        vf, vr = ppr_eth_ai(), (eth_rest.hisp / (eth_rest.hisp + eth_rest.nothisp) / SHARE[g]).values
        pf = pooled_jackknife(ETH.hisp, ETH.hisp + ETH.nothisp, SHARE[g])
        pr = pooled_jackknife(eth_rest.hisp, eth_rest.hisp + eth_rest.nothisp, SHARE[g])
        nf, nr = len(ETH), len(eth_rest)
    else:
        vf, vr = ppr_ai(g), (ai_rest[g] / ai_rest.known_sum / SHARE[g]).values
        pf = pooled_jackknife(AI[g], AI.known_sum, SHARE[g])
        pr = pooled_jackknife(ai_rest[g], ai_rest.known_sum, SHARE[g])
        nf, nr = len(AI), len(ai_rest)
    est = g not in ("AIAN", "NHPI")
    ltrows.append({
        "group": g, "trials_full": nf, "trials_reduced": nr,
        "trial_median_full": fmt3(boot_ci(vf)) if est else "Not interpreted",
        "trial_median_reduced": fmt3(boot_ci(vr)) if est else "Not interpreted",
        "pooled_full": fmt3(pf), "pooled_reduced": fmt3(pr),
        "estimable_trial_median": "yes" if est else "no"})
pd.DataFrame(ltrows).to_csv(PO("RESULTS-13-largest-trial-sensitivity.csv"), index=False)
rec["Three largest trials, participants in the analyzed race categories"] = top3.cover.sum()
rec["Three largest trials, share of participants in the analyzed race categories, %"] = (
    100 * top3.cover.sum() / AI_cov.cover.sum())

# ------------------------------------------------- period medians, as a table
PER = []
for lab, lo, hi in PERIODS:
    m = AI.start_year.between(lo, hi)
    e = ETH[ETH.nct_id.isin(AI.loc[m, "nct_id"])]
    PER.append({"Trial start period": lab, "Trials": int(m.sum()),
                **{g: fmt3(boot_ci((AI.loc[m, g + "_sh"] / SHARE[g]).values)) for g in ["Black", "White", "Asian"]},
                "Hispanic": fmt3(boot_ci(((e.hisp / (e.hisp + e.nothisp)) / SHARE["Hispanic"]).values)),
                "Trials reporting ethnicity": len(e)})
pd.DataFrame(PER).to_csv(PO("RESULTS-03-table2-period.csv"), index=False)

# ----------------------------------------------------------- geography, as a table
# The no-US-site column is a median only; the US benchmark does not apply.
LBL = {"White": "White", "Black": "Black", "Hispanic": "Hispanic or Latino", "Asian": "Asian",
       "AIAN": "American Indian or Alaska Native", "NHPI": "Native Hawaiian or Pacific Islander"}
anyus = POSTED[(POSTED.us_sites > 0) & POSTED.usable_race_ethnicity]
nous = POSTED[(POSTED.us_sites == 0) & (POSTED.n_sites > 0) & POSTED.usable_race_ethnicity]
eth_any = ETH[ETH.nct_id.isin(anyus.nct_id)]
GEO = []
for g in ["White", "Black", "Hispanic", "Asian", "AIAN", "NHPI"]:
    if g in ("AIAN", "NHPI"):
        GEO.append(dict.fromkeys(["all_US_sites_n107", "at_least_one_US_site_n111",
                                  "no_US_site_median_only_n8"], "Not estimable at this estimand"))
        GEO[-1]["group"] = LBL[g]; continue
    if g == "Hispanic":
        a = ppr_eth_ai()
        b = ((eth_any.hisp / (eth_any.hisp + eth_any.nothisp)) / SHARE[g]).values
        c = "Not reported"
    else:
        a = ppr_ai(g)
        b = ((anyus[g] / anyus[KNOWN].sum(axis=1)) / SHARE[g]).values
        c = f"{np.median((nous[g] / nous[KNOWN].sum(axis=1) / SHARE[g]).values):.2f}"
    GEO.append({"group": LBL[g], "all_US_sites_n107": fmt3(boot_ci(a)),
                "at_least_one_US_site_n111": fmt3(boot_ci(b)), "no_US_site_median_only_n8": c})
if COHORT == "all107": pd.DataFrame(GEO)[["group", "all_US_sites_n107", "at_least_one_US_site_n111",
                   "no_US_site_median_only_n8"]].to_csv(PO("RESULTS-09-geography.csv"), index=False)

# ------------------------------------------------------------ estimand comparison
# Four summaries of the same data; means are bootstrapped like medians.
def boot_ci_mean(v):
    v = np.asarray(v, dtype=float)
    draws = np.random.default_rng(SEED).choice(v, (B, len(v)), replace=True)
    s = draws.mean(axis=1)
    return float(v.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

EST = []
for g in ["White", "Black", "Asian", "AIAN", "NHPI"]:
    v, vr = ppr_ai(g), (ai_rest[g] / ai_rest.known_sum / SHARE[g]).values
    md, lo, hi = boot_ci(v); mn, mlo, mhi = boot_ci_mean(v)
    EST.append({"group": g, "trial_median": round(md, 2), "median_95CI": f"{lo:.2f}-{hi:.2f}",
                "trial_mean": round(mn, 2), "mean_95CI": f"{mlo:.2f}-{mhi:.2f}",
                "pooled_participant_weighted": round((AI[g].sum() / AI.known_sum.sum()) / SHARE[g], 2),
                "trial_median_excl_3_largest": round(float(np.median(vr)), 2),
                "pooled_excl_3_largest": round(pooled_jackknife(ai_rest[g], ai_rest.known_sum, SHARE[g])[0], 2)})
pd.DataFrame(EST).to_csv(PO("RESULTS-10-estimand.csv"), index=False)

# --------------------------------------------------------- registered conditions
# Conditions are semicolon-separated; a trial counts once per condition.
cnt = (POSTED[POSTED.usable_race_ethnicity & POSTED.nct_id.isin(AI.nct_id)].conditions.fillna("")
       .str.split(";").explode().str.strip())
cnt = cnt[cnt != ""].value_counts()
(pd.DataFrame({"condition": cnt.index, "n_usable_trials": cnt.values})
 .sort_values(["n_usable_trials", "condition"], ascending=[False, True])
 .to_csv(PO("RESULTS-12-conditions.csv"), index=False))
print(f"period, geography, estimand and condition tables written "
      f"({len(PER)}, {len(GEO)}, {len(EST)}, {len(cnt)} rows)")
pd.Series(rec, name="value").rename_axis("quantity").to_csv(PO("RESULTS-01-all-values.csv"))
