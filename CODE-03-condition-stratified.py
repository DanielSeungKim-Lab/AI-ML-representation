"""Condition groups and the condition-stratified Black comparison. Writes RESULTS-15.
M1_COHORT=category1 as in CODE-02.
"""
import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, norm

GRP = {
    "Cancer": r"cancer|carcinom|tumor|tumour|neoplas|leukemi|lymphom|myelom|melanom|oncolog",
    "Cardiovascular": r"cardiac|cardiovascular|heart|coronary|hypertens|atrial|stroke|vascular|lipid|cholesterol|arrhythm",
    "Mental health / behavioral": r"depress|anxiet|psychiat|mental|schizophren|bipolar|ptsd|substance|opioid|alcohol|smoking|addict|insomnia|sleep",
    "Neurological": r"alzheim|dementia|parkinson|epilep|neurolog|multiple sclerosis|migraine|brain injur|seizure",
    "Metabolic / endocrine": r"diabet|obes|metabolic|thyroid|weight loss",
    "Infectious": r"covid|hiv|hepatit|infect|sepsis|influenza|tubercul|vaccin|malaria",
    "Respiratory": r"asthma|copd|pulmonary|respirator|lung dis",
    "Renal / hepatic": r"kidney|renal|dialysis|liver|cirrhos",
    "Musculoskeletal / pain": r"pain|arthrit|osteo|back|fractur|knee|spine|musculoskelet",
    "Pregnancy / pediatric": r"pregnan|matern|neonat|infant|p.?ediatric|child",
}


def classify(s):
    s = (s or "").lower()
    for k, p in GRP.items():
        if re.search(p, s):
            return k
    return "Other / unclassified"


def van_elteren(groups):
    """Stratified Wilcoxon with 1/(n+1) stratum weights."""
    S = V = 0.0
    for a, c in groups:
        x = np.concatenate([a, c])
        n, n1 = len(x), len(a)
        if n1 == 0 or len(c) == 0:
            continue
        r = rankdata(x)
        w = 1.0 / (n + 1)
        S += w * (r[:n1].sum() - n1 * (n + 1) / 2.0)
        _, cnt = np.unique(x, return_counts=True)
        tie = 1 - (cnt ** 3 - cnt).sum() / float(n ** 3 - n) if n > 1 else 1.0
        V += (w ** 2) * n1 * len(c) * (n + 1) / 12.0 * tie
    z = S / np.sqrt(V)
    return z, 2 * norm.sf(abs(z))


import os
COHORT = os.environ.get("M1_COHORT", "all107")
SUFFIX = "" if COHORT == "all107" else "-category1"
d1 = pd.read_csv("DATA-01-ai-cohort.csv")
if COHORT == "category1":
    role = pd.read_csv("DATA-08-ai-role-classification.csv")
    d1 = d1[d1.nct_id.isin(role[role.final_category == 1].nct_id)]
d4 = pd.read_csv("DATA-04-comparator-counts.csv")
cc = pd.read_csv("DATA-07-comparator-conditions.csv")
shr = pd.read_csv("DATA-06-census-adult-shares.csv").set_index("group").pop_share_adult.to_dict()

ai = d1[d1.usable_race_ethnicity].copy()
ai["cg"] = ai.conditions.fillna("").map(classify)
cm = d4[d4.usable_race].copy()
cm["cg"] = cm.nct_id.map(cc.set_index("nct_id").conditions).fillna("").map(classify)
N_AI, N_CM = len(ai), len(cm)

ppr = lambda df, g: (df[g] / df.known_sum / shr[g]).values
order = (ai.cg.value_counts().reindex(list(GRP) + ["Other / unclassified"])
         .fillna(0).sort_values(ascending=False).index.tolist())

rows = []
for g in order:
    a, c = ai[ai.cg == g], cm[cm.cg == g]
    if len(a) == 0 and len(c) == 0:
        continue
    r = {"Condition group": g, "ai_n": len(a), "cmp_n": len(c),
         "ai_pct": 100 * len(a) / len(ai), "cmp_pct": 100 * len(c) / len(cm)}
    if len(a) >= 5 and len(c) >= 5:
        pa, pc = ppr(a, "Black"), ppr(c, "Black")
        r.update(ai_black=np.median(pa), cmp_black=np.median(pc),
                 p=mannwhitneyu(pa, pc, alternative="two-sided").pvalue)
    rows.append(r)

out = pd.DataFrame(rows)
assert out.ai_n.sum() == N_AI and out.cmp_n.sum() == N_CM
out.to_csv(f"RESULTS-15-condition-stratified{SUFFIX}.csv", index=False)

z, p = van_elteren([(ppr(ai[ai.cg == g], "Black"), ppr(cm[cm.cg == g], "Black"))
                    for g in out["Condition group"]])
sub = out.dropna(subset=["ai_black"])
print(f"van Elteren z={z:.3f} P={p:.4f}")
print(f"AI/ML median higher in {(sub.ai_black > sub.cmp_black).sum()} of {len(sub)} strata")
