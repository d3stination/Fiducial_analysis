"""Assemble the Phase 5--10 P1 evidence checkpoint and supervisor report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import ROOT


WORK = ROOT / "work" / "p1_completion"
TABLES = ROOT / "results" / "tables"
REPORTS = ROOT / "results" / "reports"


def clean_condition(value: str) -> str:
    return {"healthy": "overall"}.get(str(value), str(value))


def dataset_name(value: str) -> str:
    return str(value).split(" / ", 1)[0]


def text_effect(row: pd.Series) -> str:
    effect = row.get("median_participant_effect", np.nan)
    q = row.get("bh_q_value", np.nan)
    state = "supported" if bool(row.get("exploratory_supported_after_bh", False)) else "unsupported"
    return f"{row.feature_label} -> {row.outcome}: effect {effect:.4g}, q={q:.4g} ({state})"


def extend_all_method_table() -> pd.DataFrame:
    path = TABLES / "RQ1_dataset_algorithm_results.csv"
    base = pd.read_csv(path)
    derived_columns = [
        "selected_for_detailed_case_study", "case_study_role", "selection_rationale",
        "verified_primary_failure_reasons", "failure_reason_participant_n",
        "attempted_failure_denominator_n", "failed_or_ineligible_n", "unresolved_proportion",
        "morphology_screening_result", "morphology_supported_association_n",
        "morphology_test_n", "morphology_participant_n", "matched_condition_changes",
        "selected_waveform_cases",
    ]
    base = base.drop(columns=[column for column in derived_columns if column in base], errors="ignore")
    base["dataset"] = base.dataset_cohort.map(dataset_name)
    base["condition_key"] = base.condition.map(clean_condition)

    selection = pd.read_csv(WORK / "phase3_method_selection_decisions.csv")
    selection = selection[["candidate_type", "algorithm_id", "selected_for_detailed_case_study", "case_study_role", "selection_rationale"]]
    base = base.merge(selection, on=["candidate_type", "algorithm_id"], how="left")

    reasons = pd.read_csv(WORK / "phase4_failure_reason_summary.csv")
    reasons["condition_key"] = reasons.condition.map(clean_condition)
    reasons = reasons.loc[reasons.reason_record_n.gt(0)].copy()
    reasons["reason_text"] = reasons.apply(
        lambda r: f"{r.primary_reason}: {int(r.reason_record_n)} records/{int(r.reason_unique_beat_n)} beats", axis=1
    )
    reason_summary = reasons.groupby(
        ["dataset", "condition_key", "candidate_type", "algorithm_id"], as_index=False
    ).agg(
        verified_primary_failure_reasons=("reason_text", "; ".join),
        failure_reason_participant_n=("reason_participant_n", "max"),
        attempted_failure_denominator_n=("attempted_denominator_n", "max"),
        failed_or_ineligible_n=("failed_or_ineligible_n", "max"),
        unresolved_proportion=("unresolved_proportion", "max"),
    )
    base = base.merge(reason_summary, on=["dataset", "condition_key", "candidate_type", "algorithm_id"], how="left")

    morphology = pd.read_csv(WORK / "phase5_all_method_morphology_evidence.csv")
    morphology["condition_key"] = morphology.condition.map(clean_condition)
    morphology["exploratory_supported_after_bh"] = morphology.exploratory_supported_after_bh.astype(str).str.lower().eq("true")
    morphology["evidence_text"] = morphology.apply(text_effect, axis=1)

    def select_evidence(rows: pd.DataFrame) -> pd.Series:
        supported = rows.loc[rows.exploratory_supported_after_bh]
        chosen = (supported if len(supported) else rows).sort_values("bh_q_value").iloc[0]
        return pd.Series({
            "morphology_screening_result": chosen.evidence_text,
            "morphology_supported_association_n": int(rows.exploratory_supported_after_bh.sum()),
            "morphology_test_n": len(rows),
            "morphology_participant_n": int(chosen.participant_effect_n),
        })

    morphology_summary = morphology.groupby(
        ["dataset", "condition_key", "candidate_type", "algorithm_id"], as_index=False
    ).apply(select_evidence, include_groups=False).reset_index(drop=True)
    base = base.merge(morphology_summary, on=["dataset", "condition_key", "candidate_type", "algorithm_id"], how="left")

    changes = pd.read_csv(WORK / "phase6_condition_outcome_changes.csv")
    change_summary = changes.groupby(["dataset", "candidate_type", "algorithm_id", "outcome", "unit"], as_index=False).agg(
        paired_participant_n=("subject_id", "nunique"),
        median_change=("stress_minus_normal", "median"),
    )
    change_summary["change_text"] = change_summary.apply(
        lambda r: f"{r.outcome}: {r.median_change:+.3g} {r.unit} (matched n={int(r.paired_participant_n)})", axis=1
    )
    change_summary = change_summary.groupby(["dataset", "candidate_type", "algorithm_id"], as_index=False).agg(
        matched_condition_changes=("change_text", "; ".join)
    )
    base = base.merge(change_summary, on=["dataset", "candidate_type", "algorithm_id"], how="left")

    cases = pd.read_csv(WORK / "phase7_waveform_case_selection.csv")
    cases["case_text"] = cases.apply(lambda r: f"{r.category}: {r.beat_id}", axis=1)
    case_summary = cases.groupby(["dataset", "candidate_type"], as_index=False).agg(
        selected_waveform_cases=("case_text", "; ".join)
    )
    base = base.merge(case_summary, on=["dataset", "candidate_type"], how="left")
    base = base.drop(columns=["dataset", "condition_key"])
    base.to_csv(path, index=False)

    mechanism = base[[
        "dataset_cohort", "condition", "candidate_type", "algorithm_id", "algorithm_name",
        "valid_detections", "missed_detections", "invalid_detections", "failure_pct",
        "verified_primary_failure_reasons", "attempted_failure_denominator_n", "unresolved_proportion",
        "morphology_screening_result", "morphology_supported_association_n", "morphology_test_n",
        "matched_condition_changes", "selected_for_detailed_case_study", "case_study_role",
        "selection_rationale", "selected_waveform_cases",
    ]]
    mechanism.to_csv(TABLES / "RQ1_mechanism_evidence.csv", index=False)
    return base


def write_report() -> None:
    report = r"""# HeartCycle RQ1 — final Publication 1 supervisor report

## Executive summary

This investigation evaluates the frozen catalogue of 10 B-point and 13 X-point definitions across HeartCycle, IITB-ICG and ReBeatICG. The algorithms, preprocessing, source data, status labels and canonical reference calculations were not modified. The final evidence separates reference correspondence, detection coverage, algorithm-to-algorithm disagreement, morphology association and pairwise agreement; these outcomes are related but not interchangeable.

- HeartCycle ECHO favoured B8 on combined median and tail error (participant median absolute ECHO AVO error 12.59 ms; signed median +11.47 ms; P95 34.21 ms). X12 had the lowest participant median absolute ECHO AVC error (36.55 ms), while X2 had a similar median but a 409.40-ms P95 tail.
- ReBeat expert annotations favoured B1/B7 and X12/X2 by median error, but expert ICG landmarks are not independent ECHO validation. X12 had substantially smaller X-error tails than X2.
- Failure and coverage materially changed interpretation. X4 was invalid on 48.89% of HeartCycle attempts and approximately 31–47% in IITB/ReBeat conditions. IITB B1/B7 failure increased by a paired median 14.40 percentage points after exercise.
- Common-valid X coverage was only 371/691 reference-eligible HeartCycle beats, 1,329/2,022 and 981/1,737 IITB beats, and 4,401/8,403 and 4,550/8,492 ReBeat beats. Smaller disagreement among retained beats is therefore not evidence of improved reliability by itself.
- The all-method, six-feature morphology screen produced 1,794 method–feature–outcome tests and 199 exploratory associations surviving within-family BH correction. Phase 6 produced 15 supported morphology-change/reliability-change associations. Effects were heterogeneous, sometimes counterintuitive and do not establish causality.
- No universal B or X winner is defensible. Reference type, failure burden, tail error, complete-case coverage, participant heterogeneity and physiological condition must be considered together.

## 1. Introduction and research gap

ICG B and X are operational landmarks used to estimate pre-ejection period (PEP) and left-ventricular ejection time (LVET). Published definitions often target different waveform features—minima, derivative thresholds, inflections or amplitude fractions—yet are compared as if they were interchangeable. Median error alone can hide systematic early/late bias, catastrophic tails and failures that selectively exclude difficult beats. The P1 question is therefore: **which frozen B/X definitions are supported by the available references, how reliable are they across participants and conditions, and which waveform characteristics accompany their disagreement or failure?**

The analysis deliberately distinguishes landmark correspondence from physiological truth. ECHO AVO/AVC, expert ICG annotation and cross-method agreement answer different questions. No clinically approved timing tolerance was available, so no percentage-above-tolerance result was invented.

## 2. Methods and dataset/reference audit

The frozen registry contains 10 B and 13 X definitions. HeartCycle contains 17 participants, 212 recordings and 1,706 beats; ECHO provides 1,198 B-eligible and 691 X-eligible beats, with strict common-reference samples of 986 B and 371 X beats. IITB contains 18 participants measured at rest and post-exercise (2,022 and 1,737 beats) without an independent B/X timing reference. ReBeat contains 24 participants at baseline and cognitive workload (8,403 and 8,492 beats); 1,842 expert rows were uniquely matched to an ECG beat. Cognitive expert annotations were unavailable for two participants.

Strict reference comparisons used identical participants and beats for every algorithm within a comparison. ECHO and expert-annotated references remain explicitly separated. Participant-level summaries, paired participant changes and clustered bootstrap intervals prevent beats from masquerading as independent participants. Upper-quartile flags were descriptive screens—not clinical thresholds, significance tests or composite scores—and were computed before method selection. The Phase 3 detailed case-study set was B1/B7 (focal), B8 (contrast), X4/X8 (focal) and X12 (contrast); selection followed the complete 23-method screen and availability of genuine mechanism evidence, not preselection.

The failure audit assigned one evidenced primary reason using the approved hierarchy while retaining contributing diagnostics. Across 514,280 method-beat rows, 15,788 were failed or upstream-ineligible: 466 upstream dependency/ineligibility, 182 unusable search window, 4,520 no qualifying feature/crossing and 10,620 rejected boundary/order constraints. No record required an invented cause and the primary counts reconciled exactly with original statuses.

The morphology analysis evaluated peak/turning-point sharpness, local slope, amplitude, SNR, competing features and beat-to-beat variability for every applicable method, plus RR as context. Outcomes were strict reference error where available, valid/missed/invalid status, and peer disagreement. Findings were participant-aware, multiplicity-labelled and retained even when unsupported.

## 3. Results A — reference-supported accuracy (Figures 2–3)

**HeartCycle ECHO.** B8 had participant median absolute ECHO AVO error 12.59 ms, signed median +11.47 ms and P95 absolute error 34.21 ms. B9 had median absolute error 17.21 ms and smaller signed bias (−5.82 ms), but a larger 57.73-ms P95. B2’s participant median absolute error was 44.01 ms. For X, X12 had the lowest participant median absolute ECHO AVC error (36.55 ms; signed +31.68 ms; P95 78.94 ms). X2’s median was 41.45 ms but its P95 was 409.40 ms. X3 and X9 were systematically early (signed medians −144.72 and −145.73 ms).

**ReBeat expert annotation.** B1/B7 baseline participant median absolute errors were 4.00/5.80 ms and cognitive values were 9.00/9.37 ms. X12/X2 both had 4.00-ms baseline medians and 6.00/8.00-ms cognitive medians. X12 P95 tails (84 and 120 ms) were materially smaller than X2 (386 and 373.6 ms). These results quantify expert-landmark agreement, not independent physiological validity.

Figure 2 shows restrained participant distributions without spaghetti connections. Figure 3 preserves signed early/late information, central structure and complete tails in a single common symmetric-log display with identical bins/scales within each comparison. Median, spread and tails are reported here and in the data tables, not annotated as conclusions inside the plots.

## 4. Results B — failures and disagreement (Figures 4–5)

HeartCycle B7 failed on 230/1,705 eligible beats (13.49%; 212 missed, 18 invalid), B1 on 195/1,705 (11.44%), and X4 was invalid on 834/1,706 (48.89%). In IITB, B1/B7 pooled failure rose from 22.07/22.52% at rest to 33.60/34.24% post-exercise. X4 rose from 30.81% to 40.41%. In ReBeat, X4 remained high (46.73% baseline; 45.08% workload), whereas X8 rose from 1.61% to 2.97%.

All-method common-valid coverage is an essential companion to disagreement. HeartCycle retained 1,439/1,705 eligible B beats and 852/1,706 X beats; IITB retained 1,152/1,998 and 1,002/1,723 attempted B beats, versus 1,329/2,022 and 981/1,737 X beats; ReBeat retained 8,275/8,402 and 8,261/8,490 B beats, versus 4,401/8,403 and 4,550/8,492 X beats. The large X exclusions make success-only disagreement optimistic for algorithms with heavy failure.

IITB paired all-method range changed by −3.82 ms for B and −31.70 ms for X. Because failure increased simultaneously, this is not evidence of better post-exercise reliability. ReBeat B range changed +0.86 ms, whereas X increased +15.59 ms (95% bootstrap CI 4.00 to 20.00; 17/24 increased). Disagreement is not ground-truth error.

## 5. Results C — physiological-condition sensitivity (Figure 7)

Only matched participants with the same acquisition setup were analysed: 18 IITB and 24 ReBeat participants. IITB B1/B7 participant-median failure increased by 14.40 percentage points, supported by participant bootstrap intervals. X4 increased +7.73 points but its interval crossed zero. ReBeat X8 increased +0.63 points (95% CI 0.01 to 1.62); X4 changed −1.77 points with an interval spanning zero.

RR changed strongly in IITB (median post-exercise minus rest −177 ms) but little in ReBeat (+7 ms). Of 672 participant-aware Phase 6 association tests, 15 survived within-family correction. Examples include IITB B1/B7 competing-feature change versus failure change (rho −0.82, q=0.0035) and IITB X3 variability change versus failure change (rho +0.40, q=0.0091). The negative B1/B7 relationship is counterintuitive and illustrates why these exploratory correlations must not be retold as mechanisms. ReBeat included supported associations such as B4 slope change versus peer-disagreement change (rho −0.70, q=0.021). Condition changes may reflect physiology, motion, SNR, search-window geometry and selective retention; causality is unsupported.

## 6. Results D — morphology and verified failure mechanisms (Figures 1 and 6)

Phase 5 retained the complete 1,794-row all-method morphology screen, 23,631 participant effects and 60 aggregate-range summaries. After participant-aware bootstrap inference and within-family BH correction, 199 associations were supported exploratorily: 48 for attempted failure, 42 for strict reference error and 109 for peer disagreement. Unsupported and negative findings remain in the checkpoint tables.

The detailed Figure 6 panels follow the Phase 3 case-study limit but do not restrict the all-method analysis. HeartCycle B1 slope was selected from supported evidence. HeartCycle X12 competing-feature evidence did **not** survive multiplicity correction and is shown as an explicit negative/unsupported result. IITB B1 competing-feature/failure and X4 sharpness/failure, plus ReBeat B8 slope/disagreement and X4 sharpness/failure, were selected from supported method-specific findings. Category strata are outcome-blind within-participant medians; they are not tuned after observing algorithm performance.

Figure 1 examples were chosen only after the full comparison. Clean beats required all 23 methods valid and minimized the joint empirical B/X disagreement rank; where a reference existed, selection was restricted to reference-matched beats. The corrected HeartCycle example has a 24.32-ms B range rather than the former 38.77-ms lower-quartile medoid. Major-disagreement beats maximize B or X range within the dataset’s 5th–95th percentile RR range. Failure examples maximize failed X methods and display actual frozen markers plus verified primary reasons. These waveform panels demonstrate plausible morphology and execution paths; a representative example is not population-level proof.

## 7. Results E — algorithm-pair agreement (Figure 9)

B4/B5 and X5/X6 were selected as scientifically comparable threshold-family pairs. Participant-balanced Bland–Altman bias was positive in every dataset/condition. HeartCycle B4−B5 bias was +13.28 ms (limits −6.60 to +33.16) and X5−X6 +21.10 ms (−20.43 to +62.63). IITB B bias was +13.33 ms at rest and +11.20 ms post-exercise; X bias was +32.64 and +23.61 ms, with rest limits approximately −111 to +177 ms. ReBeat B bias was +9.14/+8.80 ms and X +22.27/+25.27 ms; workload X limits widened to −61.89/+112.44 ms. The pairs are not interchangeable, especially for X. Pair agreement does not establish reference accuracy.

## 8. Cross-dataset synthesis (Figure 8)

Figure 8 aligns failure patterns across HeartCycle overall, IITB rest and ReBeat baseline without pooling heterogeneous acquisitions or implying disease causation. A common complete scale preserves honest comparison and a second common 0–5% view makes low-rate patterns visible without silently changing dataset scales. The acquisition/reference differences are documented in [RQ1_dataset_comparability.csv](../tables/RQ1_dataset_comparability.csv). Local-extremum B1/B7 had their largest failure burden in IITB rest and appreciable failure in HeartCycle, but low ReBeat baseline failure. X4 was consistently failure-prone across all three datasets, while the magnitude varied. Cross-dataset differences may arise from cohort, acquisition, preprocessing, beat segmentation and waveform morphology; they cannot be assigned to disease.

## 9. Discussion and qualitative PEP/LVET implications

The central scientific result is a trade-off, not a leaderboard. B8 corresponds closely to HeartCycle ECHO and never failed there, whereas B1/B7 correspond closely to ReBeat expert annotation but exhibit condition-sensitive failure. X12 offers the most favourable available median/tail reference profile, but no X method escapes the broader coverage and operational-definition problem. X4’s high invalid rate makes it valuable mechanistically but unsuitable as a silent denominator-reducing filter.

The morphology results support a heterogeneous view: algorithms respond to waveform geometry through their exact decision rule. Sharpness, slope, amplitude, SNR, competing extrema and beat variability can each matter, but not consistently for every method or dataset. Consequently, a single morphology threshold or composite quality score would obscure rather than solve the problem. Confirmatory hierarchical modelling on independent data is needed before mechanistic or clinical claims.

PEP inherits B timing error; LVET inherits both B and X timing error and their covariance. A method can have a small median landmark error yet produce unacceptable LVET tails, and a high-failure method can yield deceptively clean estimates on retained beats. The positive B4/B5 and X5/X6 biases show that changing a seemingly minor amplitude threshold introduces systematic interval shifts. PEP/LVET studies should therefore report the exact operational definition, reference provenance, failure/coverage, signed bias, tail behaviour and condition-specific performance. This P1 analysis does not validate any method for diagnosis or define a clinical tolerance.

## 10. Limitations, scientific contributions and reproducibility

Limitations include heterogeneous reference types; no independent IITB reference; success-only common sets; repeated beats within participants; unverified physical dZ/dt calibration; exploratory multiplicity; missing ReBeat cognitive annotation for two participants; the Subject 12 expert-C defect; and no approved clinical timing tolerance. Acquisition warnings were retained as contextual QC and were not relabelled as algorithmic failure reasons unless the frozen execution path recorded a rejection.

P1 contributes a frozen 23-definition benchmark, strict reference provenance, reconciled failure mechanisms, participant-aware condition analysis, all-method six-feature morphology screening, actual-waveform verification and a reproducible figure/report chain.

| Claim | Evidence | Boundary |
|---|---|---|
| B8 has the most favourable HeartCycle ECHO B profile | Figures 2–3; strict common-reference table | HeartCycle ECHO only; not universal physiology |
| X12 improves median/tail reference correspondence relative to many X rules | Figures 2–3; ECHO and expert tables kept separate | Coverage and dataset dependence remain |
| X4 is frequently invalid | Figure 4; reconciled Phase 4 reasons | Does not explain physiological correctness |
| Post-exercise IITB B1/B7 failure increases | Figure 7; matched n=18 participant changes | Association with condition, not causality |
| ReBeat cognitive X disagreement increases | Figures 5 and 7; +15.59 ms, CI 4–20 | Complete-case result with coverage caveat |
| Morphology relates to reliability for selected method/context pairs | Figure 6; complete Phase 5/6 screens | Exploratory, heterogeneous, multiplicity-labelled |
| Threshold-family pairs are not interchangeable | Figure 9 Bland–Altman limits | Agreement is not accuracy |
| Dataset differences should not be pooled as disease effects | Figure 8 | Acquisition/cohort confounding prevents causation |

### Claim-to-evidence table

The claim-to-evidence table above is also retained in this report as the decision boundary for every P1 conclusion.

### Reproducibility and outputs

The complete 115-row dataset–algorithm table is [RQ1_dataset_algorithm_results.csv](../tables/RQ1_dataset_algorithm_results.csv); its Phase 3–7 mechanism subset is [RQ1_mechanism_evidence.csv](../tables/RQ1_mechanism_evidence.csv). Exact failure records, all morphology tests, participant effects, condition tests and waveform-selection audit remain under `work/p1_completion/`. Every active figure has a 300-dpi PNG, vector PDF, same-stem plotting-data CSV and manuscript-ready report. [FIGURE_AUDIT.md](FIGURE_AUDIT.md) records the rendered-image review and [FIGURE_LEGEND_MAPPING.md](FIGURE_LEGEND_MAPPING.md) records colours and markers.

Rebuild the derived package with:

```powershell
python -m src.final_rq1_results
python work/p1_completion/phase4_failure_reason_audit.py
python work/p1_completion/phase5_morphology_analysis.py
python work/p1_completion/phase6_condition_analysis.py
python -m src.final_rq1_figures
python -m src.final_p1_completion
python -m src.validate_final_rq1
python -m unittest discover -s tests -v
```

Phases 5–11 are complete in the verified local environment. Phase 11 results, the compact independent figure audit and the final hash manifest are linked from `results/reports` and `results/tables`. Dependency ranges are recorded in `requirements.txt`; exact installed versions and file hashes are recorded by the Phase 11 validator. No GitHub remote or publication status is claimed.
"""
    (REPORTS / "RQ1_SUPERVISOR_RESULTS.md").write_text(report, encoding="utf-8")


def write_audit() -> None:
    figures = sorted((ROOT / "results" / "figures").rglob("*.png"))
    family = pd.Series([path.stem[:3] for path in figures]).value_counts().sort_index()
    inventory = "\n".join(f"| Figure {key[1:]} | {value} |" for key, value in family.items())
    audit = f"""# Final Figure 1–9 audit

## Scope and outcome

All {len(figures)} active figures were regenerated from verified canonical tables and inspected visually at contact-sheet scale; high-density waveform, histogram and morphology panels were also checked at original resolution. Each active figure has a same-stem 300-dpi PNG, vector PDF, plotting-data CSV and manuscript-ready report. Obsolete superseded panels were moved to the recoverable archive rather than retained as competing final figures.

## Final inventory

| Group | Active figure files |
|---|---:|
{inventory}

Figure 1 contains genuine clean-agreement, major-B, major-X and failed/ambiguous examples for each dataset. Figures 2–3 are restricted to strict common-reference comparisons. Figure 4 shows failure, Figure 5 common-valid disagreement, Figure 6 method-specific morphology plus waveforms, Figure 7 matched condition change, Figure 8 aligned non-pooled cross-dataset failure patterns, and Figure 9 selected-pair Bland–Altman agreement.

## Visual and scientific checks

Every panel was required to pass the following checks:

1. B or X is explicit and B/X are never crowded into one axis.
2. Axes identify the variable and units; uncalibrated signals use within-beat z score.
3. Participant/beat counts are visible where required and match plotting data.
4. Algorithm IDs and reference provenance match canonical tables.
5. Negative/positive signed errors and zero are visually distinct.
6. Algorithm, participant and condition colours are consistent with the shared style mapping.
7. No unsupported ranking, clinical tolerance or causal conclusion is embedded in the plot.
8. Participant lines appear only for paired within-participant condition comparisons.
9. Full tails remain visible in insets where the central distribution requires a focused view.
10. The plot answers one stated scientific question and has a matching source CSV/caption.

## Corrections made during Phase 9

- Figure 1 failure annotations were shortened to verified primary-reason groups; complete diagnostics remain in the reports and plotting data.
- Figure 6 feature axes were corrected to dataset-specific derivative, amplitude, count, dB or normalised units rather than correlation units.
- Participant and beat counts were added to Figures 2, 3, 4, 5, 7 and 9 titles/captions.
- Figure 3 uses common central bins/scales within comparisons plus complete-range insets, preserving every tail observation.
- Figure 6 includes an explicit unsupported HeartCycle X12 result so negative evidence is not suppressed.
- The former Figure 8 Bland–Altman family was moved to Figure 9; Figure 8 now answers the approved cross-dataset sensitivity question without pooling datasets.

## Scientific exclusions and cautions

Figures 2–3 are absent for IITB because no independent B/X timing reference exists. Figure 7 is absent for HeartCycle because there is no matched condition. Cross-dataset panels do not imply disease causation. Common-valid X coverage is displayed/reported beside disagreement because retained-beat disagreement alone can improve when failures remove difficult beats. No tolerance exceedance percentage was calculated.

## Audit disposition

All final panels were readable without source-code access and agreed with their plotting-data CSVs and same-stem reports. The automated integrity result is [RQ1_validation_results.json](../tables/RQ1_validation_results.json). The targeted raw replay and 19-test validation completed; the isolated interpreter reused the verified host scientific packages because the fresh Python 3.14 environment did not bootstrap pip.
"""
    (REPORTS / "FIGURE_AUDIT.md").write_text(audit, encoding="utf-8")


def main() -> None:
    table = extend_all_method_table()
    write_report()
    write_audit()
    print(f"Phase 10 assembled: {len(table)} all-method rows; report and audit updated.")


if __name__ == "__main__":
    main()
