"""Phase 2: frozen raw-signal replay, participant-aware reliability and morphology."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import spearmanr

from .data_loading import ROOT, load_config
from .morphology_analysis import _legacy_api, _standardized_shape

KEYS = ['dataset_id', 'subject_id', 'recording_id', 'beat_id']
SETTINGS=json.loads((ROOT/'configs/phase2.json').read_text())
FEATURES = {'sharpness': r'Normalized curvature ($s^{-2}$)', 'slope': r'Normalized slope ($s^{-1}$)',
            'amplitude': 'Region amplitude (device units)', 'snr': 'Smooth/residual ratio (dB)',
            'competing': 'Competing extrema (count)', 'variability': 'Adjacent-beat NRMSE (a.u.)'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def heartcycle_hashes():
    # Final RQ1 deliberately replaces the figures/reports after archiving them.
    # Preserve the inherited scientific source tables, not the presentation layer.
    source_names = {
        'master_beats', 'detection_status', 'valid_detections', 'failed_detections',
        'reference_eligibility', 'reference_errors', 'participant_summary', 'morphology_features',
    }
    paths = [ROOT/'results/tables'/kind/f'{kind}_{name}.csv' for kind in ('B','X') for name in source_names]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def median_ci(values, seed=SETTINGS['seed']):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    estimate = float(np.median(values))
    if len(values) < 3:
        return estimate, np.nan, np.nan
    boot = np.median(np.random.default_rng(seed).choice(values, (SETTINGS['bootstrap_replicates'],len(values)), replace=True), axis=1)
    return estimate, float(np.quantile(boot,.025)), float(np.quantile(boot,.975))


def paired(frame, conditions, value, extra=()):
    wide = frame.pivot(index=['subject_id', *extra], columns='condition', values=value)
    wide = wide.reindex(columns=conditions).dropna().reset_index()
    wide['change'] = wide[conditions[1]] - wide[conditions[0]]
    return wide


def feature_values(signal, fs, r, c, nr, previous_shape):
    beat = signal[r:nr]
    shape = _standardized_shape(beat, 200)
    variability = float(np.sqrt(np.mean((shape-previous_shape)**2))) if previous_shape is not None and np.isfinite(shape).all() and np.isfinite(previous_shape).all() else np.nan
    scale = float(np.std(beat))
    window = min(int(round(.051*fs)) | 1, (len(beat)-1) | 1)
    smooth = savgol_filter(beat,window,3) if window>=5 and window<len(beat) else np.full_like(beat,np.nan)
    noise = np.sqrt(np.mean((beat-smooth)**2))
    power = np.std(smooth)
    snr = float(20*np.log10(power/noise)) if noise>0 and power>0 else np.nan
    normalized = (beat-np.mean(beat))/scale if scale>0 else np.full_like(beat,np.nan)
    d1 = np.gradient(normalized,1/fs)
    d2 = np.gradient(d1,1/fs)
    result = []
    for kind,left,right in [('B',r,c),('X',c,nr)]:
        row = {'candidate_type':kind, 'snr':snr, 'variability':variability,
               'sharpness':np.nan, 'slope':np.nan, 'amplitude':np.nan, 'competing':np.nan,
               'region_start_sample':left, 'region_stop_sample_exclusive':right,
               'feature_status':'UNAVAILABLE_REGION'}
        if c is not None and r<=left<right<=nr and right-left>=5 and scale>0:
            region=signal[left:right]
            a,b=left-r,right-r
            row.update(sharpness=float(np.quantile(np.abs(d2[a:b]),.95)),
                       slope=float(np.quantile(np.abs(d1[a:b]),.95)),
                       amplitude=float(np.ptp(region)),
                       competing=len(find_peaks(region)[0])+len(find_peaks(-region)[0]),
                       feature_status='MEASURED')
        result.append(row)
    return result,shape


def replay_sources():
    """Recompute R/C and frozen candidates from raw files; compare existing lineage."""
    config=load_config()
    root,_,loader,preprocess,_ = _legacy_api(config)
    from src.variability_analysis.manifest import dataset_manifest
    from src.variability_analysis.signal_processing import detect_c_point
    from src.variability_analysis.phase3 import _segment_row
    from src.variability_analysis.phase4 import _signals, _ineligible, _validate_candidate, _codes
    from src.variability_analysis.bx_candidates import DETECTORS
    from src.variability_analysis.phase4_finalize import _finalize_weissler
    cfg=json.loads((ROOT/'configs/phase2.json').read_text())
    ids={v for d in cfg['datasets'].values() for v in d['ids']}
    registry=pd.read_csv(ROOT/'configs/canonical_method_registry.csv').fillna('')
    assert registry.candidate_type.value_counts().to_dict()=={'X':13,'B':10}
    source_beats=pd.read_csv(config['sources']['phase3_beats'],low_memory=False).query('dataset_id in @ids')
    source_detections=pd.read_csv(config['sources']['phase4_candidates'],low_memory=False).query('dataset_id in @ids')
    masters=[]; detections=[]; features=[]; inventory=[]; signals={}; fingerprints={}
    for spec in dataset_manifest(root):
        if spec.dataset_id not in ids: continue
        for path in spec.discover(root):
            recording=loader(spec,path,root)
            if recording.validation_errors(): raise ValueError(recording.validation_errors())
            processed=preprocess(recording)
            bx=_signals(recording)
            fs=recording.sampling_rate_hz
            signals[recording.recording_id]=(recording,processed)
            fingerprints[str(path)]=sha(path)
            old=source_beats.loc[source_beats.recording_id.eq(recording.recording_id)].sort_values('beat_number')
            if not np.array_equal(processed.r_indices[:-1],old.beat_start_r_sample_inclusive.to_numpy(int)):
                raise ValueError(f'R replay differs: {path.name}')
            if not np.array_equal(processed.r_indices[1:],old.beat_end_next_r_sample_exclusive.to_numpy(int)):
                raise ValueError(f'Next-R replay differs: {path.name}')
            previous_shape=None
            for number,(r,nr) in enumerate(zip(processed.r_indices[:-1],processed.r_indices[1:]),1):
                r,nr=int(r),int(nr)
                cp=detect_c_point(processed,r,nr)
                c=cp.sample_index
                prior=old.loc[old.beat_number.eq(number)].iloc[0]
                if (c is not None and c!=prior.c_sample) or (c is None and pd.notna(prior.c_sample)):
                    raise ValueError(f'C replay differs: {path.name} beat {number}')
                issue=';'.join([*recording.warnings,*([cp.reason] if cp.reason else [])])
                row=dict(dataset_id=spec.dataset_id,cohort=spec.cohort,condition=spec.condition,
                         subject_id=recording.subject_id,recording_id=recording.recording_id,
                         beat_id=f'{spec.dataset_id}|{recording.recording_id}|b{number:06d}',beat_number=number,
                         source_file=str(path),r_sample=r,next_r_sample=nr,c_sample=c if c is not None else '',
                         r_timestamp_s=recording.time_s[r],next_r_timestamp_s=recording.time_s[nr],
                         c_timestamp_s=cp.time_s if c is not None else '',sampling_frequency_hz=fs,
                         sampling_frequency_source=recording.sampling_rate_source,
                         detection_status='DETECTED' if c is not None else 'C_DETECTION_FAILED',
                         qc_status='PASS' if cp.status=='VALID' else 'FLAGGED',issue_codes=issue)
                beat,_=_segment_row(row,{'sample_count':len(recording.time_s)})
                masters.append(beat)
                frows,previous_shape=feature_values(processed.analysis_dzdt,fs,r,c,nr,previous_shape)
                for f in frows: features.append({**{k:beat[k] for k in KEYS},'condition':spec.condition,**f})
                for method in registry.itertuples(index=False):
                    eligibility=beat[f'{method.candidate_type.lower()}_fiducial_eligibility']
                    eligible=eligibility.startswith('ELIGIBLE') and c is not None
                    candidate=DETECTORS[method.candidate_name](bx,r,c,nr) if eligible else _ineligible(eligibility)
                    validation,reason=_validate_candidate(candidate,method.candidate_type,bx,r,c if c is not None else r,nr)
                    qc='INVALID' if candidate.status!='DETECTED' or validation=='INVALID' else ('FLAGGED' if beat['overall_qc_status']!='PASS' else 'PASS')
                    detections.append({**{k:beat[k] for k in KEYS},'cohort':spec.cohort,'condition':spec.condition,
                        'algorithm_id':method.algorithm_id,'candidate_name':method.candidate_name,'candidate_type':method.candidate_type,
                        'candidate_sample_index':candidate.sample_index,'candidate_timestamp_s':candidate.timestamp_s,
                        'detection_status':candidate.status,'qc_status':qc,
                        'qc_issue_codes':_codes(beat['phase2_issue_codes'],beat['phase3_issue_codes'],candidate.reason,reason),
                        'r_sample':r,'c_sample':c,'next_r_sample':nr,'search_start_sample':candidate.search_start_sample,
                        'search_stop_sample_exclusive':candidate.search_stop_sample_exclusive,
                        'associated_region':'R_to_C' if method.candidate_type=='B' else 'C_to_next_R',
                        'timestamp_representation':candidate.interpolation,'method_variant':method.method_variant,
                        'dependency_candidate_name':method.dependency_candidate_name})
            inventory.append(dict(dataset_id=spec.dataset_id,condition=spec.condition,subject_id=recording.subject_id,
                recording_id=recording.recording_id,source_file=str(path),sampling_rate_hz=fs,samples=len(recording.time_s),
                duration_s=float(recording.time_s[-1]-recording.time_s[0]),detected_r_peaks=len(processed.r_indices),
                beats=max(0,len(processed.r_indices)-1),annotation_events={k:len(v.values) for k,v in recording.references.items()},
                metadata=recording.metadata,warnings=list(recording.warnings),finite_signals=bool(np.isfinite(recording.ecg).all() and np.isfinite(recording.icg).all())))
            print(f'Raw replay verified: {spec.dataset_id} / {path.name} ({len(old)} beats)',flush=True)
    master=pd.DataFrame(masters)
    for col in ['beat_start_timestamp_s','beat_end_timestamp_s','beat_duration_ms','c_timestamp_s','sampling_frequency_hz']:
        master[col]=pd.to_numeric(master[col],errors='coerce')
    status=pd.DataFrame(detections)
    # Reproduce the inherited CSV serialization boundary before X10 reads B8.
    # Without it, exact half-sample formula ties can choose the other sample.
    status['candidate_timestamp_s']=status.candidate_timestamp_s.round(9)
    status=_finalize_weissler(status,master,root)
    verify_candidates(status,source_detections)
    for file in (ROOT/'src/variability_analysis').glob('*.py'): fingerprints[str(file)]=sha(file)
    for file in [ROOT/'configs/canonical_method_registry.csv',Path(config['sources']['phase3_beats']),Path(config['sources']['phase4_candidates'])]:
        fingerprints[str(file)]=sha(file)
    return master,status,pd.DataFrame(features),inventory,signals,registry,fingerprints


def verify_candidates(status,source_detections):
    """Verify classes/samples exactly; inherited serialized times within 51 ns."""
    compare=source_detections.set_index(KEYS+['candidate_name']).sort_index()
    new=status.set_index(KEYS+['candidate_name']).sort_index()
    assert new.index.equals(compare.index)
    for col in ['detection_status','qc_status']:
        if not new[col].equals(compare[col]):
            raise ValueError(f'Frozen detection replay mismatch in {col}: {(new[col]!=compare[col]).sum()}')
    for col in ['candidate_sample_index','search_start_sample','search_stop_sample_exclusive']:
        if not np.allclose(new[col],compare[col],rtol=0,atol=0,equal_nan=True):
            raise ValueError(f'Frozen sample/window replay mismatch: {col}')
    if not np.allclose(new.candidate_timestamp_s,compare.candidate_timestamp_s,atol=5.1e-8,rtol=0,equal_nan=True):
        raise ValueError('Frozen timing replay mismatch beyond inherited export precision')


def dataset_tables(master,status,features,settings,kind):
    ids=settings['ids']
    beat=master.loc[master.dataset_id.isin(ids)].copy()
    d=status.loc[status.dataset_id.isin(ids)&status.candidate_type.eq(kind)].copy()
    methods=d.algorithm_id.drop_duplicates().tolist()
    d=d.merge(beat[KEYS+['beat_start_timestamp_s','beat_duration_ms','phase2_qc_status','combined_issue_codes']],on=KEYS,validate='many_to_one')
    d['upstream_ineligible']=d.detection_status.eq('INELIGIBLE')
    d['dependency_unavailable']=d.detection_status.str.startswith('UNAVAILABLE_REQUIRED')
    d['attempted']=~d.upstream_ineligible
    d['valid']=d.detection_status.eq('DETECTED')&d.qc_status.isin(['PASS','FLAGGED'])
    d['invalid']=d.detection_status.eq('DETECTED')&d.qc_status.eq('INVALID')
    d['missed']=d.attempted&~d.detection_status.eq('DETECTED')
    d['failure_stage']=np.select([d.upstream_ineligible,d.dependency_unavailable,d.invalid,d.missed],['UPSTREAM_REGION','REQUIRED_DEPENDENCY','BX_VALIDITY','BX_NO_CANDIDATE'],default='NONE')
    d['analysis_status']=np.select([d.upstream_ineligible,d.valid,d.missed,d.invalid],['INELIGIBLE','VALID','MISSED','INVALID'],default='UNCLASSIFIED')
    d['failure_reason']=np.where(d.valid,'',d.qc_issue_codes.fillna(''))
    d['candidate_time_from_r_ms']=1000*(d.candidate_timestamp_s-d.beat_start_timestamp_s)
    d['time_unit']='ms'; d['time_origin']='detected ECG R'; d['reference_status']='NOT_USED_FOR_RELIABILITY'
    beat['upstream_ineligible']=~beat[f'{kind.lower()}_fiducial_eligibility'].str.startswith('ELIGIBLE')
    beat['upstream_c_flagged']=beat.phase2_qc_status.eq('FLAGGED')
    beat['rr_interval_ms']=beat.beat_duration_ms
    beat['reference_status']='NO_VALIDATED_BX_TIMING_REFERENCE' if settings['ids'][0].startswith('iitb') else 'EXPERT_ANNOTATIONS_EXIST_COORDINATE_PROVENANCE_UNRESOLVED_NOT_USED'
    wide=d.loc[d.valid].pivot(index=KEYS+['condition'],columns='algorithm_id',values='candidate_time_from_r_ms').reindex(columns=methods).dropna()
    range_values=(wide.max(axis=1)-wide.min(axis=1)).rename('range_ms').reset_index()
    disagreements=[]
    for method in methods:
        group=(wide[method]-wide.drop(columns=method).median(axis=1)).abs().rename('absolute_peer_difference_ms').reset_index()
        group['algorithm_id']=method
        disagreements.append(group)
    disagreement=pd.concat(disagreements).merge(range_values,on=KEYS+['condition'],validate='many_to_one')
    disagreement=disagreement.merge(d[KEYS+['algorithm_id','candidate_name','candidate_time_from_r_ms']],on=KEYS+['algorithm_id'],validate='one_to_one')
    summary=d.groupby(['subject_id','condition','algorithm_id','candidate_name'],sort=False).agg(
        total_beats=('beat_id','size'),eligible_attempted_beats=('attempted','sum'),valid=('valid','sum'),missed=('missed','sum'),invalid=('invalid','sum'),upstream_ineligible=('upstream_ineligible','sum'),dependency_unavailable=('dependency_unavailable','sum'),recordings=('recording_id','nunique')).reset_index()
    for col in ['missed','invalid']:
        summary[f'{col}_pct']=100*summary[col]/summary.eligible_attempted_beats.replace(0,np.nan)
    summary['failure_pct']=summary.missed_pct+summary.invalid_pct
    ranges=range_values.groupby(['subject_id','condition']).range_ms.agg(common_valid_beats='size',median_range_ms='median',p95_range_ms=lambda x:x.quantile(.95)).reset_index()
    peers=disagreement.groupby(['subject_id','condition','algorithm_id']).absolute_peer_difference_ms.median().rename('median_peer_difference_ms').reset_index()
    summary=summary.merge(ranges,on=['subject_id','condition'],how='left').merge(peers,on=['subject_id','condition','algorithm_id'],how='left')
    summary['common_valid_beats']=summary.common_valid_beats.fillna(0).astype(int)
    summary['excluded_from_common_beats']=summary.total_beats-summary.common_valid_beats
    condition=[]
    for (cond,method),rows in summary.groupby(['condition','algorithm_id'],sort=False):
        value,lo,hi=median_ci(rows.failure_pct)
        condition.append(dict(condition=cond,algorithm_id=method,candidate_name=rows.candidate_name.iloc[0],participants=rows.subject_id.nunique(),
            eligible_attempted_beats=int(rows.eligible_attempted_beats.sum()),valid=int(rows.valid.sum()),missed=int(rows.missed.sum()),invalid=int(rows.invalid.sum()),upstream_ineligible=int(rows.upstream_ineligible.sum()),
            participant_median_failure_pct=value,ci_low=lo,ci_high=hi,participant_median_peer_difference_ms=rows.median_peer_difference_ms.median(),
            common_valid_beats=int(rows.common_valid_beats.sum())))
    morphology=features.loc[features.dataset_id.isin(ids)&features.candidate_type.eq(kind)].merge(beat[KEYS+['rr_interval_ms','upstream_ineligible','upstream_c_flagged']],on=KEYS,validate='one_to_one')
    morphology=morphology.merge(range_values,on=KEYS+['condition'],how='left',validate='one_to_one')
    for feature in FEATURES:
        threshold=morphology.groupby('subject_id')[feature].transform('median')
        morphology[f'{feature}_participant_median']=threshold
        morphology[f'{feature}_category']=np.where(morphology[feature].isna(),'unavailable',np.where(morphology[feature]<=threshold,'low','high'))
    return {f'{kind}_master_beats.csv':beat,f'{kind}_detection_status.csv':d,f'{kind}_valid_detections.csv':d.loc[d.valid],
            f'{kind}_failed_detections.csv':d.loc[d.missed|d.invalid],f'{kind}_condition_summary.csv':pd.DataFrame(condition),
            f'{kind}_algorithm_disagreement.csv':disagreement,f'{kind}_participant_summary.csv':summary,f'{kind}_morphology_features.csv':morphology}


def morphology_effects(tables,kind,conditions,method):
    features=tables[f'{kind}_morphology_features.csv']
    status=tables[f'{kind}_detection_status.csv']
    failures=status.loc[status.attempted&status.algorithm_id.eq(method),KEYS+['missed','invalid']].copy()
    failures['failure_pct']=100*(failures.missed|failures.invalid).astype(int)
    data=features.merge(failures,on=KEYS,how='left',validate='one_to_one')
    effects=[]; summaries=[]; associations=[]
    for f in FEATURES:
        for outcome in ['failure_pct','range_ms']:
            part=data.loc[data[f'{f}_category'].ne('unavailable')].groupby(['subject_id','condition',f'{f}_category'])[outcome].agg('mean' if outcome=='failure_pct' else 'median').rename('value').reset_index()
            part=part.rename(columns={f'{f}_category':'category'}); part['feature']=f; part['outcome']=outcome
            summaries.append(part)
            for condition,rows in part.groupby('condition'):
                wide=rows.pivot(index='subject_id',columns='category',values='value').reindex(columns=['low','high']).dropna()
                delta=wide.high-wide.low
                est,lo,hi=median_ci(delta)
                effects.append(dict(feature=f,outcome=outcome,condition=condition,paired_participants=len(wide),high_minus_low=est,ci_low=lo,ci_high=hi,positive=int((delta>0).sum()),negative=int((delta<0).sum())))
            feature_part=data.groupby(['subject_id','condition'])[f].median().rename('feature_value').reset_index()
            outcome_part=data.groupby(['subject_id','condition'])[outcome].agg('mean' if outcome=='failure_pct' else 'median').rename('outcome_value').reset_index()
            fp=paired(feature_part,conditions,'feature_value').rename(columns={'change':'feature_change'})
            op=paired(outcome_part,conditions,'outcome_value').rename(columns={'change':'outcome_change'})
            joined=fp[['subject_id','feature_change']].merge(op[['subject_id','outcome_change']],on='subject_id').dropna()
            rho=float(spearmanr(joined.feature_change,joined.outcome_change).statistic) if len(joined)>=3 and joined.feature_change.nunique()>1 and joined.outcome_change.nunique()>1 else np.nan
            for row in joined.to_dict('records'): associations.append(dict(feature=f,outcome=outcome,**row,spearman_rho_descriptive=rho))
    return pd.DataFrame(effects),pd.concat(summaries,ignore_index=True),pd.DataFrame(associations)
