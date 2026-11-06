"""Registry harvest and extraction rules (ClinicalTrials.gov API v2), the screen of
all AI/ML registrations (DATA-13) and the Census adult shares. Writes HARVEST-*.csv.
A new run returns the current registry, not the deposited cohort, and is not needed
to reproduce the results. Requires a Census API key in the CENSUSAPI environment
variable.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CT = "https://clinicaltrials.gov/api/v2/studies?"
AI_TERMS = '"artificial intelligence" OR "machine learning" OR "deep learning"'
POOL_FILTER = ("AREA[StudyType]INTERVENTIONAL AND AREA[LocationCountry]United States "
               "AND AREA[StartDate]RANGE[2015-01-01,MAX]")

RACE_KEYS = ["White", "Black", "Asian", "AIAN", "NHPI", "Multiple", "Other"]
KNOWN = RACE_KEYS
OMB5 = {"AIAN", "Asian", "Black", "NHPI", "White"}

AMBIGUOUS = r"non-?\s?white|non-?\s?black|non-asian|ai/an/asian|and/or|combined to maintain|or some other race|none of the above"
UNKNOWN = (r"unknown|not reported|unreported|not provided|missing|declin|refus|prefer not|"
           r"not specified|unspecified|chose not|don't know|do not know|not disclose|no answer|"
           r"not answered|not collected|not available|not recorded|^n/?a$|no response|did not")
RACE_WORDS = {"White": r"white|caucasi|european",
              "Black": r"black|(?<!north )african|afro",
              "Asian": r"\basian|chinese|japanese|korean|filipino|vietnamese",
              "AIAN": r"american indian|alaska|native american|amerindian|north american native|first nation",
              "NHPI": r"hawaiian|pacific island|samoan|guamanian|chamorro|\bhi/pi\b|\bnhopi\b|\bnhpi\b"}


def category(label):
    """Analysis category for a posted race or ethnicity label, or None if ambiguous."""
    l = re.sub(r"^(race|ethnicity)\s*-\s*", "", (label or "").strip().lower())
    if not l or re.search(AMBIGUOUS, l):
        return None
    if re.search(UNKNOWN, l):
        return "Unknown"
    if re.search(r"asian\s*(/|or|and|&)\s*(other\s*)?pacific", l):
        return "Asian"
    if re.search(r"more than one|multi|mixed|biracial|two or more|multiple|\+ ?another race|and another race", l):
        return "Multiple"
    if re.match(r"(all )?others?\b", l):
        return "Other"
    hits = [k for k, p in RACE_WORDS.items() if re.search(p, l)]
    if len(hits) > 1:
        return "Multiple"
    if hits:
        return hits[0]
    if re.search(r"(\bnot|\bnon)[\s,-]*(of\s*)?(hispanic|latin)", l):
        return "NotHispanic"
    if re.search(r"hispanic|latin|mexican|cuban|puerto|chicano|spanish origin", l):
        return "Hispanic"
    if re.search(r"other|middle eastern|north african|arab|mena\b|mediterranean", l):
        return "Other"
    return None


def label_of(cls, cat):
    for c in [(cat.get("title") or "").strip(), (cls.get("title") or "").strip()]:
        if c:
            return c.lower()
    return ""


def baseline_groups(bm):
    groups = bm.get("groups") or []
    total = [g["id"] for g in groups
             if (g.get("title") or "").strip().lower() in ("total", "all participants", "overall")]
    return set(total) if total else {g["id"] for g in groups}


def num(v):
    s = str(v or "").replace(",", "")
    return float(s) if re.fullmatch(r"[\d\.]+", s) else 0.0


def key_of(meas, cls, cat, xw):
    lab = label_of(cls, cat)
    if lab in ("yes", "no"):
        ctx = ((cls.get("title") or "") + " " + (meas.get("title") or "")).lower()
        return ("Hispanic" if lab == "yes" else "NotHispanic") if re.search(r"hispanic|latin", ctx) else None
    return xw(lab)


def measure_cats(meas, use, xw):
    return [(key_of(meas, cls, cat, xw), sum(num(m.get("value")) for m in (cat.get("measurements") or [])
                                             if m.get("groupId") in use), label_of(cls, cat))
            for cls in meas.get("classes") or [] for cat in cls.get("categories") or []]


def pure_not_hispanic(label):
    """True if a label states only non-Hispanic ethnicity, with no race qualifier."""
    rest = re.sub(r"(\bnot|\bnon)[\s,-]*(of\s*)?(hispanic|latin\w*)", " ", label)
    rest = re.sub(r"hispanic|latin\w*|spanish|ethnicity|origin|\bor\b|\ba\b|\bor\b|[/,()\-:]", " ", rest)
    return not rest.strip()


def pick(cands):
    return next((m for m, _ in cands if re.search(r"nih|omb", (m.get("title") or "").lower())),
                cands[0][0] if cands else None)


def extract_counts(st, xw=category):
    """Race counts from the measure carrying race categories (NIH/OMB-titled first);
    ethnicity from a measure carrying only Hispanic/non-Hispanic/unknown categories,
    else from the race measure. A race measure that lists Hispanic as one category of a
    single list, with no category stating only non-Hispanic ethnicity, records each
    participant once, so its race categories count as non-Hispanic."""
    bm = (st.get("resultsSection") or {}).get("baselineCharacteristicsModule") or {}
    use = baseline_groups(bm)
    ms = [m for m in bm.get("measures") or []
          if re.search(r"race|ethnic|hispanic|latin", (m.get("title") or "").lower())]
    info = [(m, measure_cats(m, use, xw)) for m in ms]
    keys = lambda c: {k for k, _, _ in c}
    race_m = pick([(m, c) for m, c in info if keys(c) & set(RACE_KEYS)]) or next(
        (m for m, _ in info if "race" in (m.get("title") or "").lower()), None)
    eth_sep = [(m, c) for m, c in info if keys(c) & {"Hispanic", "NotHispanic"} and not keys(c) & set(RACE_KEYS)]
    eth_m = pick(eth_sep)
    rec = {k: 0.0 for k in RACE_KEYS + ["Unknown", "hisp", "nothisp", "eth_unk", "race_count_total"]}
    rec["omb_race_categories_listed"] = 0
    rec["ethnicity_from_combined_list"] = False
    for m, c in info:
        if m is race_m:
            rec["omb_race_categories_listed"] = len(keys(c) & OMB5)
            for k, n, _ in c:
                rec["race_count_total"] += n
                if k in RACE_KEYS or k == "Unknown":
                    rec[k] += n
        if m is eth_m or (eth_m is None and m is race_m):
            single = m is not eth_m and "Hispanic" in keys(c) and not any(
                k == "NotHispanic" and pure_not_hispanic(lab) for k, _, lab in c)
            rec["ethnicity_from_combined_list"] = single
            for k, n, lab in c:
                if k == "Hispanic":
                    rec["hisp"] += n
                elif k == "NotHispanic" and (m is eth_m or single or pure_not_hispanic(lab)):
                    rec["nothisp"] += n
                elif single and k in RACE_KEYS:
                    rec["nothisp"] += n
                elif (m is eth_m) or k == "Unknown":
                    rec["eth_unk"] += n
    rec["baseline_denominator"] = sum(num(d.get("value")) for d in (bm.get("denoms") or [{}])[0].get("counts", [])
                                      if d.get("groupId") in use)
    rec["known_sum"] = sum(rec[k] for k in KNOWN)
    rc = next((c for m, c in info if m is race_m), [])
    rec["unmapped_race"] = sum(n for k, n, _ in rc if k is None)
    rec["combined_race_ethnicity"] = any(k in ("Hispanic", "NotHispanic") or re.search(r"hispanic|latin", lab)
                                         for k, n, lab in rc)
    rec["asian_pacific_combined"] = any(re.search(r"asian\s*(/|or|and|&)\s*(other\s*)?pacific", lab) for k, n, lab in rc)
    return rec


def get(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.load(r)


def pages(params, fields=None, cap=40000):
    out, token = [], None
    p = dict(params, pageSize=1000)
    if fields:
        p["fields"] = fields
    while True:
        q = dict(p)
        if token:
            q["pageToken"] = token
        r = get(CT + urllib.parse.urlencode(q))
        out += r.get("studies", [])
        token = r.get("nextPageToken")
        if not token or len(out) >= cap:
            return out
        time.sleep(0.08)


def extract(studies):
    """One row per trial: race and ethnicity counts, posted denominator, sites,
    sponsor class, start year, label-mapping flags and the reconciliation check."""
    rows = []
    for st in studies:
        ps = st.get("protocolSection") or {}
        locs = [(l.get("country") or "") for l in ((ps.get("contactsLocationsModule") or {}).get("locations") or [])]
        rec = {"nct_id": ps["identificationModule"]["nctId"], "n_sites": len(locs),
               "us_sites": sum(1 for c in locs if c == "United States"),
               "start_year": pd.to_numeric(str(((ps.get("statusModule") or {}).get("startDateStruct") or {}).get("date", ""))[:4], errors="coerce"),
               "sponsor_class": ((ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}).get("class")}
        rec.update(extract_counts(st))
        rows.append(rec)
    D = pd.DataFrame(rows)
    # Usable: a known race category and a race measure within 5% of enrollment.
    D["race_reconciles"] = (D.race_count_total / D.baseline_denominator.replace(0, pd.NA)).between(0.95, 1.05)
    D["usable_race_ethnicity"] = (D.known_sum > 0) & D["race_reconciles"]
    return D


def crosswalk(studies):
    """Category assigned to every label posted in a race or ethnicity measure (DATA-14)."""
    labs = {label_of(cls, cat) for st in studies
            for m in ((st.get("resultsSection") or {}).get("baselineCharacteristicsModule") or {}).get("measures") or []
            if re.search(r"race|ethnic|hispanic|latin", (m.get("title") or "").lower())
            for cls in m.get("classes") or [] for cat in cls.get("categories") or []}
    return pd.DataFrame({"label": sorted(labs), "category": [category(l) or "" for l in sorted(labs)]})


def census_adult_shares():
    """Adults 18 and over by race alone and by Hispanic origin, ACS 2023 one-year.

    Race tables are race-alone and include Hispanic respondents, matching the
    registry's race measures. Under-18 age bands are subtracted from each table
    total.
    """
    tables = {"White": "B01001A", "Black": "B01001B", "AIAN": "B01001C",
              "Asian": "B01001D", "NHPI": "B01001E", "Hispanic": "B01001I"}
    under18 = ["003", "004", "005", "006", "018", "019", "020", "021"]
    key = os.environ["CENSUSAPI"]
    adult = {}
    for group, tbl in tables.items():
        cols = [f"{tbl}_001E"] + [f"{tbl}_{s}E" for s in under18]
        j = get("https://api.census.gov/data/2023/acs/acs1?get=" + ",".join(cols)
                + "&for=us:1&key=" + key)
        d = {h: float(v) for h, v in zip(j[0], j[1]) if h.startswith(tbl)}
        adult[group] = d[f"{tbl}_001E"] - sum(d[f"{tbl}_{s}E"] for s in under18)
    total = get("https://api.census.gov/data/2023/acs/acs1?get=B01001_001E,B01001_003E,"
                "B01001_004E,B01001_005E,B01001_006E,B01001_027E,B01001_028E,B01001_029E,"
                "B01001_030E&for=us:1&key=" + key)
    t = {h: float(v) for h, v in zip(total[0], total[1]) if h.startswith("B01001_")}
    denom = t["B01001_001E"] - sum(v for h, v in t.items() if h != "B01001_001E")
    return pd.DataFrame({"group": list(adult), "pop_share_adult": [v / denom for v in adult.values()]})


def screen(studies, extracted):
    """Results posting among all AI/ML registrations, by narrowing group and by
    status. Months since primary completion are counted to the extraction date."""
    rows = []
    for s in studies:
        ps = s["protocolSection"]
        sm, od = ps.get("statusModule") or {}, ps.get("oversightModule") or {}
        pcd = str((sm.get("primaryCompletionDateStruct") or {}).get("date") or "")
        months = ((extracted.year - int(pcd[:4])) * 12 + extracted.month
                  - (int(pcd[5:7]) if len(pcd) >= 7 else 12)) if pcd else float("nan")
        locs = (ps.get("contactsLocationsModule") or {}).get("locations") or []
        rows.append({"status": sm.get("overallStatus"), "months": months,
                     "us": any(l.get("country") == "United States" for l in locs),
                     "fda": bool(od.get("isFdaRegulatedDrug") or od.get("isFdaRegulatedDevice")),
                     "posted": bool(s.get("hasResults"))})
    S = pd.DataFrame(rows)
    done = S.status.isin(["COMPLETED", "TERMINATED"])
    late = done & (S.months >= 12)
    steps = [("All interventional AI/ML registrations", S.status.notna() | S.status.isna()),
             ("Status completed or terminated", done),
             ("...and primary completion 12+ months ago", late),
             ("...and at least one US site", late & S.us),
             ("...and FDA-regulated drug or device", late & S.us & S.fda)]
    out = [("screening step", lab, S[m]) for lab, m in steps]
    out += [("overall status", st, S[S.status == st]) for st in S.status.value_counts().index]
    return pd.DataFrame([{"breakdown": b, "category": c, "trials": len(d),
                          "with_posted_results": int(d.posted.sum()), "pct": round(100 * d.posted.mean(), 1)}
                         for b, c, d in out])


def match_comparator(pool, cohort, per_trial=5, seed=7):
    """Frequency-match on start year and enrollment band (5 per cohort trial),
    borrowing from adjacent bands within the year when a cell is short."""
    bins = [0, 100, 300, 1000, 5000, 10 ** 9]
    pool = pool.assign(sz=pd.cut(pool.enroll, bins, labels=False))
    coh = cohort.assign(sz=pd.cut(cohort.baseline_denominator, bins, labels=False))
    rng = __import__("numpy").random.default_rng(seed)
    taken, out = set(), []
    for (year, sz), n in coh.groupby(["start_year", "sz"]).size().items():
        want = int(n) * per_trial
        for cand_sz in sorted(set(pool.sz.dropna()), key=lambda b: abs(b - sz)):
            avail = pool[(pool.year == year) & (pool.sz == cand_sz)
                         & ~pool.nct_id.isin(taken)]
            if avail.empty:
                continue
            take = avail.sample(min(want, len(avail)), random_state=int(rng.integers(1 << 31)))
            taken |= set(take.nct_id)
            out.append(take)
            want -= len(take)
            if want <= 0:
                break
    return pd.concat(out, ignore_index=True) if out else pool.head(0)


if __name__ == "__main__":
    raw_ai = pages({"query.term": AI_TERMS, "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
                    "aggFilters": "results:with"})
    ai = extract(raw_ai)
    ai.to_csv(os.path.join(HERE, "HARVEST-ai-posted-results.csv"), index=False)
    reg = pages({"query.term": AI_TERMS, "filter.advanced": "AREA[StudyType]INTERVENTIONAL"},
                fields="NCTId|OverallStatus|PrimaryCompletionDate|HasResults|IsFDARegulatedDrug|"
                       "IsFDARegulatedDevice|LocationCountry")
    screen(reg, date.today()).to_csv(os.path.join(HERE, "HARVEST-ai-registry-screen.csv"), index=False)

    idx = pages({"filter.advanced": POOL_FILTER, "aggFilters": "results:with"},
                fields="NCTId|StartDate|EnrollmentCount|LeadSponsorClass|LocationCountry")
    pool = pd.DataFrame([{
        "nct_id": s["protocolSection"]["identificationModule"]["nctId"],
        "year": pd.to_numeric(str((s["protocolSection"].get("statusModule") or {})
                                  .get("startDateStruct", {}).get("date", ""))[:4], errors="coerce"),
        "enroll": ((s["protocolSection"].get("designModule") or {})
                   .get("enrollmentInfo") or {}).get("count"),
        "countries": "|".join(sorted({(l.get("country") or "") for l in (
            (s["protocolSection"].get("contactsLocationsModule") or {}).get("locations") or [])})),
    } for s in idx])
    pool = pool[pool.countries.eq("United States") & pool.enroll.notna()
                & ~pool.nct_id.isin(set(ai.nct_id))]
    pool.to_csv(os.path.join(HERE, "HARVEST-comparator-pool.csv"), index=False)

    cohort = ai[(ai.us_sites == ai.n_sites) & (ai.n_sites > 0) & ai.usable_race_ethnicity]
    matched = match_comparator(pool, cohort)
    matched.to_csv(os.path.join(HERE, "HARVEST-comparator-matched.csv"), index=False)

    # Same extraction and check; trials posting no race measure are kept for ethnicity.
    raw_cmp = pages({"filter.ids": ",".join(matched.nct_id)})
    cmp_counts = extract(raw_cmp)
    cmp_counts["race_reconciles"] = cmp_counts.race_reconciles.where(
        cmp_counts.known_sum > 0, True)
    cmp_counts["usable_race"] = (cmp_counts.known_sum > 0) & cmp_counts.race_reconciles
    cmp_counts.to_csv(os.path.join(HERE, "HARVEST-comparator-counts.csv"), index=False)

    crosswalk(raw_ai + raw_cmp).to_csv(os.path.join(HERE, "HARVEST-race-ethnicity-crosswalk.csv"), index=False)
    census_adult_shares().to_csv(os.path.join(HERE, "HARVEST-census-adult-shares.csv"), index=False)
    print(f"{len(ai)} AI/ML trials with posted results; {len(pool)} eligible comparator trials")
