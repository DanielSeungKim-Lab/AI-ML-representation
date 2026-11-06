AI/ML CLINICAL TRIAL REPRESENTATION: REPLICATION DEPOSIT

Data are released under CC BY 4.0 and code under the MIT license.

No participant-level data are included.

SOURCES

ClinicalTrials.gov API v2, retrieved October 7, 2026 (DATA-01 to 05 and 13),
October 12 (DATA-07) and November 2 (DATA-09, 10 and 12; the comparator pool is
restricted to results first posted by October 7). Baseline results for all
trials were retrieved again on November 5, 2026; race and ethnicity counts in
DATA-01 to 04 and 09, DATA-02 omb_race_categories_listed and DATA-12
intervention_model come from that retrieval.

The registry changes over time, so these files, not a new query, are the record
of the analyzed trials.

US Census Bureau American Community Survey: 2023 1-year national adult shares
(DATA-06) and 2019-2023 5-year county adult shares (DATA-11).


FILES

CODE-01  Registry harvest, extraction rules and crosswalk. Not needed to
         reproduce results.
CODE-02  Main analysis. Writes RESULTS-01 to 14; RESULTS-01 lists every value.
CODE-03  Condition-stratified analysis. Writes RESULTS-15.
CODE-04  Revision analyses. Writes RESULTS-16 to 25.

DATA-01  107 US-only AI/ML trials (analyzed cohort)
DATA-02  143 AI/ML trials with posted baseline results
DATA-03  Ethnicity counts for the 86 AI/ML trials reporting ethnicity
DATA-04  530 matched conventional trials
DATA-05  Matching strata for DATA-04
DATA-06  US adult population shares
DATA-07  Registered conditions for DATA-04
DATA-08  AI/ML role classifications by each reviewer and final
DATA-09  All 22,377 trials in the eligible comparator pool
DATA-10  US site coordinates and counties
DATA-11  County adult population shares
DATA-12  Registry design fields
DATA-13  Results posting among the 2,858 AI/ML registrations (Supplementary
         Tables 16 and 17); summary counts only, not trial-level records
DATA-14  Category assigned to every race and ethnicity label posted by the
         deposited trials


RUN

    python CODE-02-main-analysis.py
    python CODE-03-condition-stratified.py
    python CODE-04-revision-analyses.py

Run from this directory. Prefix each command with M1_COHORT=category1 for the
sensitivity analysis of the 53 trials in which AI/ML was the intervention under
evaluation; outputs then end in -category1. Prefix CODE-04 with
M1_BANDS=labeled for left-closed enrollment bands; outputs then end in
-labeled-bands. Tested with Python 3.11.15, pandas 2.3.3, NumPy 2.4.6 and
SciPy 1.17.1.


DEFINITIONS

PPR: a group's share of a trial's participants divided by its share of US
adults; 1.0 is proportional.

Race counts come from the baseline measure carrying race categories (NIH/OMB
titled first). Ethnicity comes from a measure carrying only Hispanic,
non-Hispanic and unknown categories, otherwise from the race measure; when that
measure is a single list in which Hispanic is one category, its race categories
count as non-Hispanic (ethnicity_from_combined_list in DATA-03 and 04). Labels
without a category in DATA-14 are excluded.

Race data are used only if the race measure's posted total (race_count_total)
is within 5% of posted baseline enrollment.

Denominators are participants in known categories.

Enrollment bands are right-closed (1-100, 101-300, 301-1,000, 1,001-5,000,
>5,000): posted baseline enrollment for AI/ML trials, registered enrollment for
comparator trials.

AI/ML role (DATA-08): 1, intervention under evaluation; 2, component of the
intervention; 3, analysis or measurement only; 4, no AI/ML role.
