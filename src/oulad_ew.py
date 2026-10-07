"""Early-warning data for OULAD: who is still enrolled on day k, and what was observable before day k.

The unit is an enrolment (one student in one module presentation), not a student: 28,785 students
hold 32,593 enrolments. Days are counted from the module's start (day 0), as in OULAD itself.

Rules that keep the future out of every feature:
- the at-risk set on day k is enrolments registered before day k and not unregistered before day k;
- clicks count only if dated before day k;
- an assessment counts as due only if its deadline is before day k, and a submission (and its
  score) only if it was submitted before day k. OULAD does not record when a mark was returned,
  so scores are treated as known at submission: a small optimism, stated in the notebook.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["code_module", "code_presentation", "id_student"]
PRESENTATION = ["code_module", "code_presentation"]
STATIC_CAT = ["code_module", "gender", "region", "disability"]
STATIC_NUM = ["age_ord", "edu_ord", "imd_ord", "num_of_prev_attempts", "studied_credits", "reg_lead_days"]
EDU_ORDER = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification",
             "Post Graduate Qualification"]
AGE_ORDER = ["0-35", "35-55", "55<="]
TOP_TYPES = ["homepage", "oucontent", "forumng", "quiz", "resource", "subpage", "url"]


def load(data_dir: Path) -> dict:
    read = lambda n: pd.read_csv(data_dir / (n + ".csv"), na_values="?")
    t = {n: read(n) for n in ["studentInfo", "studentRegistration", "studentAssessment", "assessments", "courses", "vle"]}
    enrol = t["studentInfo"].merge(t["studentRegistration"], on=KEYS, how="left")
    enrol = enrol.merge(t["courses"], on=PRESENTATION, how="left")
    enrol["withdrawn"] = (enrol["final_result"] == "Withdrawn").astype(int)
    enrol["age_ord"] = enrol["age_band"].map({a: i for i, a in enumerate(AGE_ORDER)})
    enrol["edu_ord"] = enrol["highest_education"].map({e: i for i, e in enumerate(EDU_ORDER)})
    # "0-10%" -> 0 ... "90-100%" -> 9 (OULAD writes one band as "10-20" without the % sign)
    enrol["imd_ord"] = enrol["imd_band"].str.extract(r"^(\d+)")[0].astype(float) / 10
    enrol["reg_lead_days"] = -enrol["date_registration"]
    t["enrol"] = enrol
    return t


def clean_enrolments(enrol: pd.DataFrame) -> tuple:
    """Drop the enrolments whose withdrawal timing is unknowable or contradictory."""
    no_date = (enrol["withdrawn"] == 1) & enrol["date_unregistration"].isna()
    contradiction = (enrol["withdrawn"] == 0) & enrol["date_unregistration"].notna()
    kept = enrol[~no_date & ~contradiction].copy()
    return kept, {"withdrawn_without_date": int(no_date.sum()), "unregistered_but_not_withdrawn": int(contradiction.sum())}


def daily_clicks(data_dir: Path, vle: pd.DataFrame) -> pd.DataFrame:
    """Clicks per enrolment per day, split by activity type; cached as parquet next to the raw file."""
    cache = data_dir / "daily_clicks.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    sv = pd.read_csv(data_dir / "studentVle.csv",
                     dtype={"code_module": "category", "code_presentation": "category", "id_student": "int32",
                            "id_site": "int32", "date": "int16", "sum_click": "int32"})
    sv = sv.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    sv["atype"] = np.where(sv["activity_type"].isin(TOP_TYPES), sv["activity_type"], "other")
    d = sv.pivot_table(index=KEYS + ["date"], columns="atype", values="sum_click", aggfunc="sum",
                       fill_value=0, observed=True).reset_index()
    d.columns.name = None
    d["clicks"] = d[TOP_TYPES + ["other"]].sum(axis=1)
    for c in PRESENTATION:
        d[c] = d[c].astype(str)
    d.to_parquet(cache, index=False)
    return d


def at_risk(enrol: pd.DataFrame, k: int) -> pd.DataFrame:
    """Enrolments registered before day k and not unregistered before day k."""
    reg_ok = enrol["date_registration"].isna() | (enrol["date_registration"] < k)
    still = enrol["date_unregistration"].isna() | (enrol["date_unregistration"] >= k)
    return enrol[reg_ok & still].copy()


def click_features(daily: pd.DataFrame, k: int) -> pd.DataFrame:
    seen = daily[daily["date"] < k]
    g = seen.groupby(KEYS)
    f = pd.DataFrame({
        "clicks_total": g["clicks"].sum(),
        "active_days": g["date"].nunique(),
        "last_active_day": g["date"].max(),
        "first_active_day": g["date"].min(),
        "clicks_precourse": seen[seen["date"] < 0].groupby(KEYS)["clicks"].sum(),
        "clicks_last14": seen[seen["date"] >= k - 14].groupby(KEYS)["clicks"].sum(),
    })
    for a in TOP_TYPES:
        f["share_" + a] = g[a].sum() / f["clicks_total"]
    return f.reset_index()


def assessment_features(t: dict, k: int) -> pd.DataFrame:
    asm = t["assessments"]
    asm = asm[asm["assessment_type"] != "Exam"]
    due = asm[asm["date"] < k][PRESENTATION + ["id_assessment"]]
    sa = t["studentAssessment"].merge(asm[PRESENTATION + ["id_assessment", "date"]], on="id_assessment")
    sub = sa[sa["date_submitted"] < k]
    n_due = due.groupby(PRESENTATION).size().rename("n_due").reset_index()
    sub_due = sub.merge(due, on=PRESENTATION + ["id_assessment"])
    g = sub.groupby(KEYS)
    f = pd.DataFrame({
        "n_submitted": g.size(),
        "mean_score": g["score"].mean(),
        "min_score": g["score"].min(),
        "n_late": sub.assign(late=sub["date_submitted"] > sub["date"]).groupby(KEYS)["late"].sum(),
        "n_banked": g["is_banked"].sum(),
        "n_due_submitted": sub_due.groupby(KEYS).size(),
    }).reset_index()
    return f, n_due


def features_at(t: dict, enrol: pd.DataFrame, daily: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per enrolment still at risk on day k, with every feature observable before day k."""
    r = at_risk(enrol, k)
    r = r.merge(click_features(daily, k), on=KEYS, how="left")
    a, n_due = assessment_features(t, k)
    r = r.merge(a, on=KEYS, how="left").merge(n_due, on=PRESENTATION, how="left")
    for c in ["clicks_total", "active_days", "clicks_precourse", "clicks_last14", "n_submitted", "n_late",
              "n_banked", "n_due_submitted", "n_due"]:
        r[c] = r[c].fillna(0)
    r["days_since_active"] = np.where(r["last_active_day"].notna(), k - r["last_active_day"], k + 30)
    r["n_missed"] = r["n_due"] - r["n_due_submitted"]
    r["log_clicks"] = np.log1p(r["clicks_total"])
    # Engagement relative to classmates in the same presentation on the same day: modules differ
    # several-fold in how much their course sites are used, so raw clicks are not comparable.
    r["log_clicks_rel"] = r["log_clicks"] - r.groupby(PRESENTATION)["log_clicks"].transform("median")
    r["k"] = k
    return r


def numeric_features(k: int) -> list:
    base = STATIC_NUM + ["log_clicks", "log_clicks_rel", "active_days", "clicks_precourse", "clicks_last14",
                         "days_since_active"] + ["share_" + a for a in TOP_TYPES]
    if k > 0:
        base += ["n_due", "n_submitted", "n_missed", "mean_score", "min_score", "n_late", "n_banked"]
    return base
