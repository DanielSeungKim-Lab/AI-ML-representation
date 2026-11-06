"""Revision analyses: repeated comparator selection, the full comparator pool,
site-county benchmark, design characteristics, overall standardized differences,
reviewer agreement and AI/ML role definitions. Writes RESULTS-16 to RESULTS-25.
M1_COHORT=category1 as in CODE-02. The 1,000 selections take about two minutes.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, norm, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(HERE, f)
COHORT = os.environ.get("M1_COHORT", "all107")
SUFFIX = ("" if COHORT == "all107" else "-category1") + ("-labeled-bands" if os.environ.get("M1_BANDS") == "labeled" else "")
RIGHT = os.environ.get("M1_BANDS") != "labeled"   # bands as implemented (right-closed) unless M1_BANDS=labeled
PO = lambda f: P(f.replace(".csv", SUFFIX + ".csv"))
DRAWS = 1000

KNOWN = ["White", "Black", "Asian", "AIAN", "NHPI", "Multiple", "Other"]
GROUPS = ["Black", "White", "Asian", "Hispanic"]
BINS = [0, 100, 300, 1000, 5000, 10 ** 9]
SEED, B = 7, 4000

AI_ALL = pd.read_csv(P("DATA-01-ai-cohort.csv"))
ETH_ALL = pd.read_csv(P("DATA-03-ai-ethnicity.csv"))
SHARE = pd.read_csv(P("DATA-06-census-adult-shares.csv")).set_index("group")["pop_share_adult"]
ROLE = pd.read_csv(P("DATA-08-ai-role-classification.csv"))
POOL = pd.read_csv(P("DATA-09-comparator-pool-22377.csv")).sort_values("nct_id").reset_index(drop=True)
ROLE_AI = ROLE.set_index("nct_id").final_category
CAT1 = set(ROLE_AI[ROLE_AI == 1].index)
CAT12 = set(ROLE_AI[ROLE_AI.isin([1, 2])].index)


def cohort(ids):
    return (AI_ALL[AI_ALL.nct_id.isin(ids)].reset_index(drop=True),
            ETH_ALL[ETH_ALL.nct_id.isin(ids)].reset_index(drop=True))


AI, ETH = cohort(CAT1 if COHORT == "category1" else set(AI_ALL.nct_id))
CMP = pd.read_csv(P("DATA-04-comparator-counts.csv"))
CMP = CMP[CMP.race_reconciles].reset_index(drop=True)
CMP_RACE = CMP[CMP[KNOWN].sum(axis=1) > 0]
CMP_ETH = CMP[(CMP.hisp + CMP.nothisp) > 0]


def ai_vec(AIc, ETHc, g):
    if g == "Hispanic":
        return pd.Series((ETHc.hisp / (ETHc.hisp + ETHc.nothisp) / SHARE[g]).values, index=ETHc.nct_id)
    return pd.Series((AIc[g + "_sh"] / SHARE[g]).values, index=AIc.nct_id)


def cmp_vec(C, g):
    """Same PPR definitions as CODE-02. C must already exclude non-reconciling trials."""
    if g == "Hispanic":
        e = C[(C.hisp + C.nothisp) > 0]
        return pd.Series((e.hisp / (e.hisp + e.nothisp) / SHARE[g]).values, index=e.nct_id)
    r = C[C[KNOWN].sum(axis=1) > 0]
    return pd.Series((r[g] / r[KNOWN].sum(axis=1) / SHARE[g]).values, index=r.nct_id)


def boot_ci(v):
    v = np.asarray(v, dtype=float)
    s = np.median(np.random.default_rng(SEED).choice(v, (B, len(v)), replace=True), axis=1)
    return float(np.median(v)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def wmedian(x, w):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    return float(x[np.searchsorted(np.cumsum(w) / w.sum(), 0.5)])


def van_elteren(a, b, sa, sb):
    """Stratified Wilcoxon rank-sum test (van Elteren) with tie correction."""
    a, b, sa, sb = map(np.asarray, (a, b, sa, sb))
    num = var = 0.0
    for s in sorted(set(sa) & set(sb)):  # fixed order so floating-point sums are identical across runs
        x, y = a[sa == s], b[sb == s]
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            continue
        rk = rankdata(np.concatenate([x, y]))
        N = m + n
        t = np.unique(rk, return_counts=True)[1]
        v = m * n / 12 * ((N + 1) - (t ** 3 - t).sum() / (N * (N - 1)))
        w = 1 / (N + 1)
        num += w * (rk[:m].sum() - m * (N + 1) / 2)
        var += w ** 2 * v
    return float(2 * norm.sf(abs(num / np.sqrt(var))))


def smd_multi(c1, c2):
    """Yang and Dalton (2012) multivariate standardized difference, category counts."""
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    keep = (c1 + c2) > 0
    p1, p2 = (c1[keep] / c1.sum())[1:], (c2[keep] / c2.sum())[1:]
    S = (np.diag(p1) - np.outer(p1, p1) + np.diag(p2) - np.outer(p2, p2)) / 2
    d = p1 - p2
    return float(np.sqrt(d @ np.linalg.pinv(S) @ d))


def strata_ai(AIc):
    return [f"{y}|{b}" for y, b in zip(AIc.start_year, pd.cut(AIc.baseline_denominator, BINS, labels=False, right=RIGHT))]


POOL["stratum"] = [f"{y}|{b}" for y, b in zip(POOL.year, pd.cut(POOL.enroll, BINS, labels=False, right=RIGHT))]
POOL_OK = POOL[POOL.race_reconciles_cmp].reset_index(drop=True)


# ------------------------------------------------- RESULTS-19 full eligible pool
def full_pool(AIc, ETHc, nboot=1000):
    """Comparator medians from every eligible pool trial in the cohort's start-year by
    enrollment-band strata, each stratum weighted to the cohort's distribution, with
    a stratified bootstrap interval and a van Elteren test stratified the same way."""
    rows, sa_all = [], strata_ai(AIc)
    for g in GROUPS:
        a = ai_vec(AIc, ETHc, g)
        sa = [dict(zip(AIc.nct_id, sa_all))[n] for n in a.index]
        c = cmp_vec(POOL_OK, g)
        sc = POOL_OK.set_index("nct_id").stratum.reindex(c.index).values
        target = pd.Series(sa).value_counts(normalize=True)
        inside = np.array([s in target.index for s in sc])
        c, sc = c[inside], sc[inside]
        size = pd.Series(sc).value_counts()
        w = np.array([target[s] / size[s] for s in sc])
        rng = np.random.default_rng(SEED)
        groups = {s: np.where(sc == s)[0] for s in size.index}
        bs = []
        for _ in range(nboot):
            idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in groups.values()])
            bs.append(wmedian(c.values[idx], w[idx]))
        rows.append({"group": g, "ai_n": len(a), "ai_median": float(np.median(a)),
                     "pool_trials_in_ai_strata": len(c), "pool_weighted_median": wmedian(c.values, w),
                     "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                     "van_elteren_p": van_elteren(a.values, c.values, sa, sc)})
    return pd.DataFrame(rows)


FP = full_pool(AI, ETH)
FP.to_csv(PO("RESULTS-19-full-pool-comparator.csv"), index=False)


# ------------------------------------------------- RESULTS-16 to 18 repeated selection
def match_comparator(pool, coh, per_trial=5, seed=7):
    """CODE-01's frequency-matched draw, unchanged."""
    pool = pool.assign(sz=pd.cut(pool.enroll, BINS, labels=False, right=RIGHT))
    coh = coh.assign(sz=pd.cut(coh.baseline_denominator, BINS, labels=False, right=RIGHT))
    rng = np.random.default_rng(seed)
    taken, out = set(), []
    for (year, sz), n in coh.groupby(["start_year", "sz"]).size().items():
        want = int(n) * per_trial
        for cand_sz in sorted(set(pool.sz.dropna()), key=lambda b: abs(b - sz)):
            avail = pool[(pool.year == year) & (pool.sz == cand_sz) & ~pool.nct_id.isin(taken)]
            if avail.empty:
                continue
            take = avail.sample(min(want, len(avail)), random_state=int(rng.integers(1 << 31)))
            taken |= set(take.nct_id)
            out.append(take)
            want -= len(take)
            if want <= 0:
                break
    return pd.concat(out, ignore_index=True)


if COHORT == "all107":
    coh = AI[["nct_id", "start_year", "baseline_denominator"]]
    pool_race = POOL[POOL.reports_race & POOL.race_reconciles_cmp].reset_index(drop=True)
    av = {g: ai_vec(AI, ETH, g).values for g in GROUPS}
    for label, pool_used, fname in [("full", POOL, "RESULTS-17-comparator-draws-full-pool.csv"),
                                    ("race", pool_race, "RESULTS-18-comparator-draws-race-reporting-pool.csv")]:
        recs = []
        for s in range(1, DRAWS + 1):
            d = match_comparator(pool_used, coh, seed=s)
            C = d[d.race_reconciles_cmp]
            r = {"draw": s, "n_drawn": len(d), "n_retained": len(C),
                 "n_race": int((C[KNOWN].sum(axis=1) > 0).sum()), "n_eth": int(((C.hisp + C.nothisp) > 0).sum())}
            for g in GROUPS:
                cv = cmp_vec(C, g).values
                r[g + "_median"], r[g + "_p"] = float(np.median(cv)), float(mannwhitneyu(av[g], cv).pvalue)
            recs.append(r)
        D = pd.DataFrame(recs)
        D.to_csv(PO(fname), index=False)
        globals()["DRAWS_" + label] = D
    rows = []
    for label, D in [("Full eligible pool", DRAWS_full), ("Race-reporting pool", DRAWS_race)]:
        for g in GROUPS:
            m = D[g + "_median"]
            rows.append({"pool": label, "group": g, "ai_median": float(np.median(av[g])),
                         "cmp_median_median": float(m.median()), "cmp_median_p2_5": float(m.quantile(0.025)),
                         "cmp_median_p97_5": float(m.quantile(0.975)),
                         "pct_ai_above_cmp": float(100 * (np.median(av[g]) > m).mean()),
                         "pct_p_below_0_05": float(100 * (D[g + "_p"] < 0.05).mean()),
                         "p_median": float(D[g + "_p"].median()), "p_p2_5": float(D[g + "_p"].quantile(0.025)),
                         "p_p97_5": float(D[g + "_p"].quantile(0.975)),
                         "primary_selection_percentile": float(100 * (m < float(np.median(cmp_vec(CMP, g)))).mean())
                         if label == "Full eligible pool" else None})
    pd.DataFrame(rows).to_csv(PO("RESULTS-16-comparator-resampling.csv"), index=False)


# ------------------------------------------------- RESULTS-20 site-county benchmark
SITES = pd.read_csv(P("DATA-10-us-site-locations.csv"))
CS = pd.read_csv(P("DATA-11-county-adult-shares-acs5-2023.csv")).set_index("fips")


def catchment(ids):
    """Adult population shares of the counties hosting a trial's US sites, each
    distinct site county weighted equally."""
    s = SITES[SITES.nct_id.isin(ids)].drop_duplicates(["nct_id", "county_fips"])
    sh = s.join(CS[[g + "_share" for g in ["White", "Black", "AIAN", "Asian", "NHPI", "Hispanic"]]], on="county_fips")
    return sh.groupby("nct_id")[[c for c in sh.columns if c.endswith("_share")]].mean()


CA, CC = catchment(AI.nct_id), catchment(CMP.nct_id)
rows = []
for g in GROUPS:
    a, c = ai_vec(AI, ETH, g) * SHARE[g], cmp_vec(CMP, g) * SHARE[g]
    a, c = a[a.index.isin(CA.index)], c[c.index.isin(CC.index)]
    ka, kc = CA.loc[a.index, g + "_share"], CC.loc[c.index, g + "_share"]
    fm = lambda t, k=1: f"{k * t[0]:.{2 if k == 1 else 1}f} ({k * t[1]:.{2 if k == 1 else 1}f}-{k * t[2]:.{2 if k == 1 else 1}f})"
    pa, pc = (a / ka).values, (c / kc).values
    rows.append({"group": g, "ai_n": len(a), "cmp_n": len(c), "national_share_pct": round(100 * SHARE[g], 3),
                 "ai_catch_share_pct": fm(boot_ci(ka), 100), "ai_catch_ppr": fm(boot_ci(pa)),
                 "ai_expected_national_ppr": round(float(np.median(ka)) / SHARE[g], 2),
                 "cmp_catch_share_pct": fm(boot_ci(kc), 100), "cmp_catch_ppr": fm(boot_ci(pc)),
                 "cmp_expected_national_ppr": round(float(np.median(kc)) / SHARE[g], 2),
                 "p_catch_share": float(mannwhitneyu(ka, kc).pvalue), "p_catch_ppr": float(mannwhitneyu(pa, pc).pvalue)})
R20 = pd.DataFrame(rows)
R20.to_csv(PO("RESULTS-20-site-catchment.csv"), index=False)
n_cty = SITES.drop_duplicates(["nct_id", "county_fips"]).groupby("nct_id").size()
SINGLE_COUNTY = (float((n_cty.reindex(AI.nct_id) == 1).mean()), float((n_cty.reindex(CMP_RACE.nct_id) == 1).mean()))


# ------------------------------------------------- RESULTS-21 and 22 design characteristics
DES = pd.read_csv(P("DATA-12-design-features.csv"), keep_default_na=False).set_index("nct_id")
LEVELS = {"itype": ["Drug or biological", "Behavioral", "Device", "Other"],
          "allocation": ["Randomized", "Not applicable (single group)", "Non-randomized"],
          "masking": ["None (open label)", "Masked"], "phase": ["NA", "Phase 1-4"],
          "sites": ["Single US site", "Multiple US sites"],
          "primary_purpose": ["Treatment", "Prevention", "Health services research", "Other", "Basic science",
                              "Supportive care", "Diagnostic", "Screening", "Not recorded"]}
ab, cb = ai_vec(AI, ETH, "Black"), cmp_vec(CMP, "Black")
da, dc = DES.loc[ab.index], DES.loc[cb.index]
rows, strat = [], []
for var, levels in LEVELS.items():
    for lv in levels:
        x, y = ab[da[var] == lv], cb[dc[var] == lv]
        r = {"variable": var, "level": lv, "ai_n": len(x), "ai_pct": 100 * len(x) / len(ab),
             "cmp_n": len(y), "cmp_pct": 100 * len(y) / len(cb)}
        if len(x) and len(y):
            r.update(ai_black=float(np.median(x)), cmp_black=float(np.median(y)), p=float(mannwhitneyu(x, y).pvalue))
        rows.append(r)
    strat.append({"variable": var,
                  "overall_smd": smd_multi([(da[var] == l).sum() for l in levels], [(dc[var] == l).sum() for l in levels]),
                  "stratified_black_p": van_elteren(ab.values, cb.values, da[var].values, dc[var].values)})
sg_a, sg_c = da[da.allocation == "Not applicable (single group)"], dc[dc.allocation == "Not applicable (single group)"]
x, y = ab[sg_a.index[sg_a.intervention_model == "SINGLE_GROUP"]], cb[sg_c.index[sg_c.intervention_model == "SINGLE_GROUP"]]
rows.append({"variable": "allocation not applicable, registered single-group model", "level": "Single group",
             "ai_n": len(x), "cmp_n": len(y), "ai_black": float(np.median(x)), "cmp_black": float(np.median(y)),
             "p": float(mannwhitneyu(x, y).pvalue)})
for m in sorted(set(sg_a.intervention_model) | set(sg_c.intervention_model)):
    rows.append({"variable": "intervention_model (allocation not applicable)", "level": m,
                 "ai_n": int((sg_a.intervention_model == m).sum()), "cmp_n": int((sg_c.intervention_model == m).sum())})
pd.DataFrame(rows).to_csv(PO("RESULTS-21-design-features.csv"), index=False)
pd.DataFrame(strat).to_csv(PO("RESULTS-22-design-stratified.csv"), index=False)


# ------------------------------------------------- RESULTS-23 overall standardized differences
def period(y):
    if y < 2015:
        return "Before 2015"
    return "2015-2017" if y < 2018 else ("2018-2019" if y < 2020 else ("2020-2021" if y < 2022 else "2022-2025"))


MATCH = pd.read_csv(P("DATA-05-comparator-matching.csv")).set_index("nct_id")
cm_r = CMP_RACE.set_index("nct_id")
cov = {"Trial start period": (AI.start_year.map(period), cm_r.year.map(period)),
       "Enrollment band": (pd.cut(AI.baseline_denominator, BINS, labels=False, right=RIGHT),
                           MATCH.sz.reindex(cm_r.index)),
       "Sponsor class": (AI.sponsor_class, cm_r.sponsor)}
srows = []
for k, (x, y) in cov.items():
    lv = sorted(set(x.astype(str)) | set(y.astype(str)))
    srows.append({"covariate": k, "overall_smd": smd_multi([(x.astype(str) == l).sum() for l in lv],
                                                           [(y.astype(str) == l).sum() for l in lv]),
                  "levels": len(lv), "ai_total": len(x), "cmp_total": len(y)})
pd.DataFrame(srows).to_csv(PO("RESULTS-23-overall-smd.csv"), index=False)


# ------------------------------------------------- RESULTS-24 agreement between the two reviewers
def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po, pe = (a == b).mean(), sum((a == v).mean() * (b == v).mean() for v in np.union1d(a, b))
    return float((po - pe) / (1 - pe))


def role_agreement(R):
    a, b, f = R.category_A.values, R.category_B.values, R.final_category.values
    rows = [("Same category", int((a == b).sum()), kappa(a, b)),
            ("Same answer on whether AI/ML was received (categories 1-2 vs 3-4)", int(((a <= 2) == (b <= 2)).sum()), kappa(a <= 2, b <= 2)),
            ("Disagreements between categories 1 and 2", int(((a != b) & (np.minimum(a, b) == 1) & (np.maximum(a, b) == 2)).sum()), None),
            ("Reviewed by the investigator (final category 3 or 4, or a disagreement)", int(((f >= 3) | (a != b)).sum()), None)]
    return pd.DataFrame([{"measure": m, "trials": n, "of": len(R), "kappa": k} for m, n, k in rows])


if COHORT == "all107" and RIGHT:   # band-independent; written by the primary run only
    role_agreement(ROLE).to_csv(P("RESULTS-24-role-agreement.csv"), index=False)


# ------------------------------------------------- RESULTS-25 AI/ML role definitions
def compare(AIc, ETHc, C, label, comparator):
    out = []
    for g in GROUPS:
        a, c = ai_vec(AIc, ETHc, g).values, cmp_vec(C, g).values
        m, lo, hi = boot_ci(a)
        out.append({"ai_definition": label, "comparator": comparator, "group": g, "ai_n": len(a), "ai_median": m,
                    "ai_lo": lo, "ai_hi": hi, "cmp_n": len(c), "cmp_median": float(np.median(c)),
                    "p": float(mannwhitneyu(a, c).pvalue)})
    return out


if COHORT == "all107":
    CAT34 = set(ROLE_AI[ROLE_AI.isin([3, 4])].index)
    rows = []
    for ids, label in [(CAT1, "AI/ML intervention (category 1)"), (CAT12, "AI/ML intervention or component (categories 1-2)"),
                       (CAT34, "Analysis or measurement only, or no AI/ML role (categories 3-4)"), (set(AI_ALL.nct_id), "All AI/ML trials")]:
        A2, E2 = cohort(ids)
        rows += compare(A2, E2, CMP, label, "Frequency-matched comparators")
        fp = FP if label == "All AI/ML trials" else full_pool(A2, E2, nboot=200)
        for _, r in fp.iterrows():
            rows.append({"ai_definition": label, "comparator": "Full eligible pool, weighted to the definition's strata",
                         "group": r.group, "ai_n": r.ai_n, "ai_median": r.ai_median, "cmp_n": r.pool_trials_in_ai_strata,
                         "cmp_median": r.pool_weighted_median, "p": r.van_elteren_p})
    pd.DataFrame(rows).to_csv(PO("RESULTS-25-ai-role-definitions.csv"), index=False)

print(f"{COHORT}: {len(AI)} AI/ML trials, {len(CMP)} comparator trials; "
      f"{len(SITES.drop_duplicates(['lat', 'lon']))} distinct site coordinates; single-county share "
      f"{100 * SINGLE_COUNTY[0]:.1f}% vs {100 * SINGLE_COUNTY[1]:.1f}%")
