"""Reproduction comparison and evidence-based final Phase 2 audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase2_analysis import sha,heartcycle_hashes


def hashes():
    files=[p for family in ['tables','figures','reports'] for dataset in ['IITB','ReBeat'] for p in (ROOT/'results'/family/dataset).rglob('*') if p.is_file()]
    files += [ROOT/'results/reports/P1_CONSOLIDATED_RQ1.md',ROOT/'results/reports/PHASE2_LEGEND_MAPPING.md']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--rerun',action='store_true'); args=parser.parse_args()
    work=ROOT/'work/phase2'
    if args.rerun:
        before=hashes()
        run=subprocess.run([sys.executable,'src/run_phase2.py'],cwd=ROOT)
        after=hashes()
        result={'command':'python src/run_phase2.py','exit_code':run.returncode,'files_compared':len(before),'identical':before==after,'before':before,'after':after}
        (work/'reproduction_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        if run.returncode or before!=after: raise RuntimeError('Fresh reproduction did not regenerate identical outputs')
        print(f'Fresh raw rerun reproduced {len(before)} outputs byte-for-byte.',flush=True)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    log=test.stdout+test.stderr
    (work/'test_log.txt').write_text(log,encoding='utf-8')
    print(log,flush=True)
    if test.returncode: raise RuntimeError('Validation tests failed; do not label complete')
    audit=json.loads((work/'execution_audit.json').read_text())
    visual=json.loads((work/'visual_review.json').read_text()) if (work/'visual_review.json').exists() else {}
    reproduction=json.loads((work/'reproduction_check.json').read_text()) if (work/'reproduction_check.json').exists() else {}
    text=['# Phase 2 reproducibility and validation audit',
          '## Scope and preservation',
          f'HeartCycle: {len(audit["heartcycle_sha256"])} existing result files checked by SHA-256; unchanged = {heartcycle_hashes()==audit["heartcycle_sha256"]}. No HeartCycle regeneration, deletion, or result replacement was performed. The older test was scoped explicitly to B/X so new dataset audit reports do not change its expected HeartCycle count.',
          'IITB: 18 matched participants, 36 recordings, 3,759 R-defined beats (2,022 under-rest; 1,737 post-exercise). ReBeat: 24 matched participants, 48 recordings, 16,895 beats (8,403 baseline; 8,492 cognitive workload). The frozen catalogue remains exactly 10 B and 13 X methods.',
          '## Computational checks',
          'PASS — exact R/C replay, candidate sample/search-window equality, detection/QC classifications, and candidate timestamps within inherited export precision. The 51-ns serialization bound is not a scientific error tolerance. Reproducing the original nine-decimal B8 export resolves four Weissler half-sample ties without changing its formula.',
          'PASS — source IDs and matched conditions; unique beat-method keys; mutually exclusive statuses; upstream ineligibility and required dependencies; reconciled attempted denominators; strict all-method common-valid sets; beat-level range and leave-one-out peer median; participant medians/P95; outcome-blind feature categories; measurable failed-beat features; formula unit tests; figure/report pairing; 300-dpi exports; actual plotted artist values compared against CSV calculations.',
          '## Reproduction',
          f'Fresh documented raw-data command exit: {reproduction.get("exit_code","not yet checked")}. Output files compared: {reproduction.get("files_compared","pending")}. Byte-identical results: {reproduction.get("identical","pending")}. Detailed input/output hashes and complete test log are retained in work/phase2/. The original MAT files are required for a raw replay; the frozen implementation is packaged under src/variability_analysis.',
          '## Test output','```text\n'+log.strip()+'\n```',
          '## Figure inventory and rendered visual audit',
          '40 figures: per dataset and point type, F01 has three participant-matched waveform examples; F02 failure components; F03 subject ranges; F04 definition-versus-peer differences; F05A paired failure; F05B paired range; F06A morphology/failure; F06B morphology/range. Each has a matching nine-section report. No reference-error figures or heatmaps were added.',
          '| Figure | Render inspection | Notes |','|---|---|---|']
    inspected=0
    for row in audit['figures']:
        review=visual.get(row['figure'],{})
        okay=review.get('sha256')==sha(ROOT/row['figure']) and review.get('status')=='PASS'
        inspected+=okay
        text.append(f'| {row["figure"]} | {"PASS" if okay else "PENDING"} | {review.get("notes","Not certified from PNG export alone")} |')
    text += [f'Current rendered files personally inspected and checksum-matched: {inspected}/40.',
             'Review checks: correct dataset/B/X/condition; explicit axes and units; identifiable point meaning; consistent colour/shape conventions; readable method labels and legends; no shifted timing markers; no numerical result strings in plots; no misleading range-as-error claims. Coincident waveform detections are separated by method-labelled timing rows, not by changing their timestamps. Fixed horizontal offsets distinguish paired participant points without changing outcome values.',
             '## Scientific limitations and decision gate',
             'No clinical validation or universal algorithm winner is established. Local ReBeat MAT files report 250 Hz despite 2,000-Hz acquisition metadata; annotation coordinate/selection provenance remains unresolved. IITB native −dZ/dt polarity and inherited C-selection warnings constrain interpretation. Morphology summaries are operational region-based features; amplitude is uncalibrated, quality is a filtered smooth/residual proxy, extrema counts are partly coupled to some rules, and sharpness/slope are potentially redundant. Complete-case disagreement can select easier beats. Exploratory bootstrap intervals and descriptive correlations do not prove causation or control multiplicity.',
             '## Files added or changed',
             'Added src/phase2_analysis.py, src/phase2_outputs.py, src/run_phase2.py, src/validate_phase2.py, configs/phase2.json, docs/phase2_methodology.md, tests/test_phase2.py, 32 dataset-specific CSVs, 40 PNG figures and 40 figure reports, two dataset audits, the consolidated RQ1 report and Phase 2 legend/audit reports. Updated README.md, the top-level config map, and the existing test file’s HeartCycle-only path scope. No frozen source detector or HeartCycle output was changed. Existing Phase 2 versions were overwritten in place; a recoverable rendering backup is kept only in work/phase2/, not final results.']
    (ROOT/'results/reports/PHASE2_VALIDATION_AUDIT.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    print(f'Audit saved; visual review {inspected}/40.',flush=True)


if __name__=='__main__': main()
