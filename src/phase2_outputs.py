"""Dataset-specific Phase 2 figures, reports and preservation audit."""
from __future__ import annotations

import json
import textwrap
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .data_loading import ROOT
from .phase2_analysis import KEYS, FEATURES, dataset_tables, paired, median_ci, morphology_effects, heartcycle_hashes, sha
from .restructured_package import DISPLAY_NAMES

RESULTS=ROOT/'results'
SETTINGS=json.loads((ROOT/'configs/phase2.json').read_text())
COLOURS=SETTINGS['condition_colours']
PARTICIPANT_PALETTE=SETTINGS['participant_palette']


def md(frame,digits=2):
    frame=frame.copy()
    for col in frame.select_dtypes('number'): frame[col]=frame[col].round(digits)
    frame=frame.fillna('unavailable')
    lines=['| '+' | '.join(map(str,frame.columns))+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    lines+=['| '+' | '.join(str(x).replace('|',' / ') for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def report(dataset,kind,stem,question,source,counts,result,interpretation,limitation,decision,registry,axes):
    path=RESULTS/'reports'/dataset/kind/f'{stem}.md'; path.parent.mkdir(parents=True,exist_ok=True)
    methods=registry.loc[registry.candidate_type.eq(kind)]
    mapping='; '.join(f'{r.algorithm_id} ({DISPLAY_NAMES[r.algorithm_id]}) = {r.candidate_name}' for r in methods.itertuples())
    sections={'Question':question,'Dataset':f'{dataset}; verified conditions. Figure: `{stem}.png`. Source: '+source,
              'Algorithms':mapping,'Axes and units':axes,'N':counts,'Key numerical result':result,
              'Interpretation':interpretation,'Limitation':limitation,'Decision':decision}
    path.write_text('# '+stem+'\n\n'+'\n\n'.join(f'## {i}. {key}\n\n{value}' for i,(key,value) in enumerate(sections.items(),1))+'\n',encoding='utf-8')


def save(fig,dataset,kind,stem,ledger,axes):
    path=RESULTS/'figures'/dataset/kind/f'{stem}.png'; path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig)
    plotted=[]
    for ax in fig.axes:
        for artist in [*ax.lines,*ax.collections,*ax.patches]:
            if not artist.get_gid(): continue
            if hasattr(artist,'get_offsets'):
                xy=artist.get_offsets().filled(np.nan).tolist()
            elif hasattr(artist,'get_height'):
                xy=[[float(artist.get_x()+artist.get_width()/2),float(artist.get_height())]]
            else: xy=np.column_stack([artist.get_xdata(),artist.get_ydata()]).tolist()
            plotted.append({'id':artist.get_gid(),'xy':xy})
    ledger.append({'figure':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'axes':axes,'visual_review':'pending','actual_axis_labels':[[a.get_xlabel(),a.get_ylabel()] for a in fig.axes],'plotted_data':plotted})


def label_methods(ax,methods):
    ax.set_xticks(np.arange(len(methods)),[m+'\n'+textwrap.fill(DISPLAY_NAMES[m],15) for m in methods],rotation=35,ha='right')
    ax.set_xlabel('Algorithm definition')


def change_table(summary,conditions,value,methods=None):
    rows=[]
    groups=summary.groupby('algorithm_id',sort=False) if methods else [('Across-method range',summary)]
    for method,data in groups:
        pair=paired(data,conditions,value)
        estimate,lo,hi=median_ci(pair.change)
        rows.append(dict(algorithm=method,paired_participants=len(pair),median_change=estimate,CI_low=lo,CI_high=hi,
                         increased=int((pair.change>0).sum()),decreased=int((pair.change<0).sum()),unchanged=int((pair.change==0).sum())))
    return pd.DataFrame(rows)


def count_text(master,summary):
    population=master.groupby('condition').agg(participants=('subject_id','nunique'),recordings=('recording_id','nunique'),beats=('beat_id','size'),upstream_region_ineligible=('upstream_ineligible','sum'),C_flagged=('upstream_c_flagged','sum')).reset_index()
    counts=summary[['condition','algorithm_id','eligible_attempted_beats','valid','missed','invalid','upstream_ineligible','dependency_unavailable','common_valid_beats','excluded_from_common_beats']].groupby(['condition','algorithm_id']).sum().reset_index()
    return md(population,0)+'\n\n'+md(counts,0)


def condition_figures(dataset,kind,tables,settings,registry,ledger):
    master=tables[f'{kind}_master_beats.csv']; status=tables[f'{kind}_detection_status.csv']
    summary=tables[f'{kind}_participant_summary.csv']
    methods=registry.loc[registry.candidate_type.eq(kind),'algorithm_id'].tolist()
    conditions=settings['conditions']; labels=settings['labels']
    subjects=sorted(master.subject_id.unique()); colour=dict(zip(subjects,PARTICIPANT_PALETTE))
    counts=count_text(master,summary)
    failure_changes=change_table(summary,conditions,'failure_pct',True)
    greatest=failure_changes.loc[failure_changes.median_change.idxmax()]
    components=[]
    for value in ['missed_pct','invalid_pct']:
        delta=paired(summary.loc[summary.algorithm_id.eq(greatest.algorithm)],conditions,value).change
        components.append(f'{value}: median paired change {delta.median():.3f} percentage points')
    failure_takeaway=(f'The largest median paired failure increase is {greatest.algorithm}: {greatest.median_change:.3f} percentage points; '
                      f'{greatest.increased} increased, {greatest.decreased} decreased, {greatest.unchanged} unchanged among {greatest.paired_participants} pairs. '
                      +'; '.join(components)+'. If this maximum is zero, no positive median increase is demonstrated. ')
    selected=failure_changes.assign(magnitude=lambda d:d.median_change.abs(),order=lambda d:d.algorithm.str[1:].astype(int)).sort_values(['magnitude','order'],ascending=[False,True]).algorithm.iloc[0]
    # F02: mean participant failure components; uncertainty around same estimator.
    fig,ax=plt.subplots(figsize=(max(10,len(methods)*.85),5.2))
    plotted=[]
    for j,condition in enumerate(conditions):
        rows=summary.loc[summary.condition.eq(condition)]
        missed=rows.groupby('algorithm_id').missed_pct.mean().reindex(methods)
        invalid=rows.groupby('algorithm_id').invalid_pct.mean().reindex(methods)
        x=np.arange(len(methods))+(j-.5)*.36
        for patch,method in zip(ax.bar(x,missed,width=.33,color=COLOURS[j],edgecolor=COLOURS[j]),methods): patch.set_gid(f'missed/{condition}/{method}')
        for patch,method in zip(ax.bar(x,invalid,bottom=missed,width=.33,color=COLOURS[j],hatch='///',edgecolor='#FFFFFF',linewidth=.25),methods): patch.set_gid(f'invalid/{condition}/{method}')
        for pos,method in zip(x,methods):
            v=rows.loc[rows.algorithm_id.eq(method),'failure_pct'].dropna().to_numpy()
            boot=np.mean(np.random.default_rng(SETTINGS['seed']).choice(v,(SETTINGS['bootstrap_replicates'],len(v))),axis=1)
            lo,hi=np.quantile(boot,[.025,.975]); value=np.mean(v)
            ax.errorbar(pos,value,yerr=[[max(0,value-lo)],[max(0,hi-value)]],color='#263F4F',capsize=2,lw=.8)
            plotted.append(dict(condition=condition,algorithm_id=method,participant_mean_failure_pct=value,CI_low=lo,CI_high=hi))
    label_methods(ax,methods); ax.set_ylabel('Missed + invalid detection rate (%)')
    ax.set_title(f'{dataset} · {kind}-point detection failure')
    ax.legend(handles=[Patch(color=COLOURS[j],label=labels[j]) for j in range(2)]+[Patch(facecolor='#AAAAAA',label='Solid: missed'),Patch(facecolor='white',edgecolor='#555555',hatch='///',label='Hatched: invalid')],frameon=False,ncol=4,loc='upper left',bbox_to_anchor=(0,1.13))
    stem=f'F02_{dataset}_{kind}_condition_failure'; axes='x: algorithm definition; y: participant mean missed+invalid rate (%)'
    save(fig,dataset,kind,stem,ledger,axes)
    report(dataset,kind,stem,'Which definitions fail more, and in which condition?',f'{kind}_participant_summary.csv; {kind}_detection_status.csv',counts,
           md(pd.DataFrame(plotted))+'\n\nPaired changes (second minus first; percentage points):\n\n'+md(failure_changes),
           failure_takeaway+'Bars give equal weight to each participant. Solid and hatched components separate missed and invalid outcomes. Compare paired directions alongside the condition means.',
           'Intervals bootstrap participants within condition. C timing flags remain accepted upstream warnings; ineligible regions and required-method dependency failures are separately tabulated. No independent reference accuracy is assessed.',
           'Interpret condition sensitivity jointly with common-valid retention and upstream warnings.',registry,axes)
    # F03: participant median same-beat range; P95 stays in report.
    ranges=summary.drop_duplicates(['subject_id','condition'])
    fig,ax=plt.subplots(figsize=(max(10,len(subjects)*.42),4.5))
    pairs=paired(ranges,conditions,'median_range_ms').set_index('subject_id')
    for i,subject in enumerate(subjects):
        if subject in pairs.index:
            vals=pairs.loc[subject,conditions].to_numpy(float)
            ax.plot([i-.13,i+.13],vals,color='#9AA2A7',lw=.8)
        for j,condition in enumerate(conditions):
            v=ranges.loc[ranges.subject_id.eq(subject)&ranges.condition.eq(condition),'median_range_ms']
            if len(v): ax.scatter(i+(j-.5)*.26,v.iloc[0],color=COLOURS[j],marker=['o','s'][j],s=26).set_gid(f'range/{condition}/{subject}')
    ax.set_xticks(range(len(subjects)),[s.split(':')[-1].replace('Subject_','S') for s in subjects],rotation=45,ha='right')
    ax.set(xlabel='Participant ID',ylabel='Median within-beat algorithm timing range (ms)',title=f'{dataset} · {kind}-point participant disagreement')
    ax.legend(handles=[Line2D([],[],color=COLOURS[j],marker=['o','s'][j],ls='',label=labels[j]) for j in range(2)],frameon=False)
    stem=f'F03_{dataset}_{kind}_subject_disagreement'; axes='x: participant ID; y: participant median across-method beat range (ms)'
    save(fig,dataset,kind,stem,ledger,axes)
    range_change=change_table(ranges,conditions,'median_range_ms')
    report(dataset,kind,stem,'Does disagreement change consistently across participants?',f'{kind}_participant_summary.csv; {kind}_algorithm_disagreement.csv',counts,
           md(ranges[['subject_id','condition','common_valid_beats','median_range_ms','p95_range_ms']])+'\n\n'+md(range_change),
           'Each point summarizes ranges computed within identical complete beats; the report retains the participant P95 without mixing it into the median plot.',
           'Complete-case composition can change by condition. Larger range means definition sensitivity, not a demonstrated error.',
           'Use participant directions and retention counts to judge whether the change is broad or concentrated.',registry,axes)
    # F04: definition deviation, grouped boxes and participant observations.
    fig,ax=plt.subplots(figsize=(max(10,len(methods)*.85),5.4))
    for j,condition in enumerate(conditions):
        for i,method in enumerate(methods):
            rows=summary.loc[summary.condition.eq(condition)&summary.algorithm_id.eq(method)].sort_values('subject_id').dropna(subset=['median_peer_difference_ms'])
            x=i+(j-.5)*.36
            if len(rows):
                ax.boxplot([rows.median_peer_difference_ms],positions=[x],widths=.30,showfliers=False,patch_artist=True,boxprops={'facecolor':'white','edgecolor':COLOURS[j]},medianprops={'color':COLOURS[j]},whiskerprops={'color':COLOURS[j]},capprops={'color':COLOURS[j]})
                for off,row in zip(np.linspace(-.07,.07,len(rows)),rows.itertuples()):
                    ax.scatter(x+off,row.median_peer_difference_ms,s=11,marker=['o','s'][j],color=COLOURS[j],alpha=.7).set_gid(f'peer/{condition}/{method}/{row.subject_id}')
    label_methods(ax,methods); ax.set(ylabel='Participant median absolute peer difference (ms)',title=f'{dataset} · {kind}-point definition disagreement')
    ax.legend(handles=[Line2D([],[],color=COLOURS[j],marker=['o','s'][j],ls='',label=labels[j]) for j in range(2)],frameon=False)
    stem=f'F04_{dataset}_{kind}_definition_disagreement'; axes='x: definition; y: participant median absolute difference from other-method median (ms)'
    save(fig,dataset,kind,stem,ledger,axes)
    peer_change=change_table(summary,conditions,'median_peer_difference_ms',True)
    report(dataset,kind,stem,'Which definitions deviate most from their peers, and does this change?',f'{kind}_algorithm_disagreement.csv; {kind}_participant_summary.csv',counts,
           md(summary.groupby(['condition','algorithm_id']).median_peer_difference_ms.median().reset_index())+'\n\nPaired changes:\n\n'+md(peer_change),
           'For each complete beat, the target is excluded from the peer median. Absolute differences are summarized within each participant before comparison.',
           'Peer agreement is not ground truth. Missing methods exclude the beat for every compared definition.',
           'Identify systematic definition differences without declaring the divergent definition incorrect.',registry,axes)
    # F05: separate paired failure and range.
    for code,value,subset,ylabel in [('F05A','failure_pct',summary.loc[summary.algorithm_id.eq(selected)],'Algorithm failure rate (%)'),('F05B','median_range_ms',ranges,'Median across-method timing range (ms)')]:
        pair=paired(subset,conditions,value)
        fig,ax=plt.subplots(figsize=(8,6))
        offsets=dict(zip(subjects,np.linspace(-.065,.065,len(subjects))))
        for row in pair.itertuples(index=False):
            values=[getattr(row,c) for c in conditions]
            line,=ax.plot(np.array([0,1])+offsets[row.subject_id],values,'o-',color=colour[row.subject_id],lw=.85,ms=4,alpha=.85,label=row.subject_id.split(':')[-1].replace('Subject_','S'))
            line.set_gid(f'paired/{value}/{selected}/{row.subject_id}')
        ax.set_xticks([0,1],labels); ax.set(xlabel='Recording condition',ylabel=ylabel,title=f'{dataset} · {kind}-point paired '+('failure: '+selected+' '+DISPLAY_NAMES[selected] if code=='F05A' else 'disagreement'))
        ax.set_xlim(-.2,1.2)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=6,frameon=False,fontsize=7)
        fig.subplots_adjust(bottom=.26)
        stem=f'{code}_{dataset}_{kind}_paired_'+('failure' if code=='F05A' else 'disagreement')
        axes='x: condition; y: '+ylabel
        save(fig,dataset,kind,stem,ledger,axes)
        estimate,lo,hi=median_ci(pair.change)
        report(dataset,kind,stem,'How large is the within-participant condition change?',f'{kind}_participant_summary.csv',f'{len(pair)} matched participants.\n\n'+counts,
               f'Median paired change = {estimate:.3f}; 95% participant bootstrap interval {lo:.3f} to {hi:.3f}.\n\n'+md(pair),
               (f'{selected} was selected by largest absolute median paired failure change, ties by numeric registry ID. This is a descriptive data-selected display; all methods are retained in F02 tables. ' if code=='F05A' else '')+'Lines connect recordings from the same participant, not matched heartbeats. A small fixed participant-specific horizontal offset separates coincident points; no outcome value is jittered.',
               'Observational condition differences can reflect physiology, recording quality and complete-case selection. The selected-method interval does not adjust for selection.',
               'Report magnitude, direction and uncertainty without causal attribution.',registry,axes)
    return selected,colour,failure_changes,range_change,peer_change


def waveform_figures(dataset,kind,tables,settings,registry,signals,ledger,selected):
    status=tables[f'{kind}_detection_status.csv']; master=tables[f'{kind}_master_beats.csv']; morph=tables[f'{kind}_morphology_features.csv']
    d=master.merge(morph[KEYS+['range_ms','snr','competing','sharpness']],on=KEYS,validate='one_to_one')
    failed=status.assign(failed=lambda x:x.missed|x.invalid).groupby(KEYS).failed.sum().reset_index()
    d=d.merge(failed,on=KEYS)
    conditions=settings['conditions']; labels=settings['labels']
    style=json.loads((ROOT/'configs/figure_style.yaml').read_text())
    used=set()
    for example in ['close','ambiguous','failed']:
        pool=d.loc[~d.beat_id.isin(used)].copy()
        if example=='close':
            pool=pool.loc[pool.range_ms.notna()&~pool.upstream_c_flagged&~pool.upstream_ineligible&pool.snr.ge(pool.groupby('subject_id').snr.transform('median'))]
            ordered=pool.sort_values(['range_ms','beat_id'])
        elif example=='ambiguous': ordered=pool.loc[pool.range_ms.notna()].sort_values(['range_ms','beat_id'],ascending=[False,True])
        else: ordered=pool.loc[pool.failed.gt(0)].sort_values(['failed','beat_id'],ascending=[False,True])
        choice=None
        for _,row in ordered.iterrows():
            partner=d.loc[d.subject_id.eq(row.subject_id)&d.condition.ne(row.condition)&~d.beat_id.isin(used)]
            if len(partner):
                rr_median=partner.beat_duration_ms.median()
                partner=partner.assign(distance=(partner.beat_duration_ms-rr_median).abs()).sort_values(['distance','beat_id'])
                choice=pd.DataFrame([row,partner.iloc[0]])
                break
        if choice is None: continue
        fig,axes=plt.subplots(2,2,figsize=(12,7.2),sharex='col',gridspec_kw={'height_ratios':[2,1.7]})
        method_rows=registry.loc[registry.candidate_type.eq(kind)].reset_index(drop=True)
        details=[]
        for j,(condition,label) in enumerate(zip(conditions,labels)):
            row=choice.loc[choice.condition.eq(condition)].iloc[0]; used.add(row.beat_id)
            rec,proc=signals[row.recording_id]; r=int(row.beat_start_r_sample_inclusive); nr=int(row.beat_end_next_r_sample_exclusive)
            y=proc.analysis_dzdt[r:nr]; std=y.std(); y=(y-y.mean())/std if std>0 else y*np.nan
            time=1000*(rec.time_s[r:nr]-rec.time_s[r])
            ax=axes[0,j]; strip=axes[1,j]; ax.plot(time,y,color='#263F4F',lw=1.1)
            candidates=status.loc[status.beat_id.eq(row.beat_id)]
            missing=[]
            for c in candidates.loc[candidates.detection_status.eq('DETECTED')].itertuples():
                if pd.isna(c.candidate_timestamp_s): continue
                marker='x' if c.invalid else ('^' if kind=='B' else 'v')
                if time[0]<=c.candidate_time_from_r_ms<=time[-1]:
                    ax.scatter(c.candidate_time_from_r_ms,np.interp(c.candidate_time_from_r_ms,time,y),color=style['algorithm_colours'][c.candidate_name],marker=marker,s=28,zorder=4)
                position=method_rows.index[method_rows.algorithm_id.eq(c.algorithm_id)][0]
                strip.scatter(c.candidate_time_from_r_ms,position,color=style['algorithm_colours'][c.candidate_name],marker=marker,s=30).set_gid(f'candidate/{row.beat_id}/{c.algorithm_id}')
            for pos,m in method_rows.iterrows():
                state=candidates.loc[candidates.algorithm_id.eq(m.algorithm_id)].iloc[0]
                if state.detection_status!='DETECTED':
                    strip.text(.99,pos,state.analysis_status.lower(),transform=strip.get_yaxis_transform(),ha='right',va='center',fontsize=7,color='#555555')
            target=candidates.loc[candidates.algorithm_id.eq(selected)].iloc[0]
            if pd.notna(target.search_start_sample) and pd.notna(target.search_stop_sample_exclusive):
                ax.axvspan((target.search_start_sample-r)*1000/rec.sampling_rate_hz,(target.search_stop_sample_exclusive-r)*1000/rec.sampling_rate_hz,color='#B8C5CD',alpha=.3,label=f'{selected} actual search window')
            signal_label='−dZ/dt' if dataset=='IITB' else 'dZ/dt'
            ax.set_ylabel(f'Filtered {signal_label} (standardized a.u.)')
            ax.set_title(f'{label} · {row.subject_id.split(":")[-1]}\n{row.recording_id}\n{row.beat_id.split("|")[-1]}',loc='left',fontsize=8)
            ax.legend(frameon=False,loc='upper right',fontsize=7)
            strip.set_yticks(range(len(method_rows)),[m.algorithm_id+' '+DISPLAY_NAMES[m.algorithm_id] for m in method_rows.itertuples()],fontsize=7)
            strip.set_ylim(len(method_rows)-.5,-.5)
            strip.set_ylabel('Algorithm definition')
            strip.set_xlabel('Time relative to ECG R-peak (ms)')
            strip.spines['left'].set_visible(False)
            details.append(dict(condition=condition,subject_id=row.subject_id,recording_id=row.recording_id,beat_id=row.beat_id,range_ms=row.range_ms,failed_methods=row.failed,
                upstream_C_warning=row.upstream_c_flagged,RR_ms=row.rr_interval_ms,sharpness=row.sharpness,competing_extrema=row.competing,quality_proxy_db=row.snr,
                failure_reasons='; '.join(c.algorithm_id+': '+str(c.failure_reason) for c in candidates.loc[candidates.missed|candidates.invalid].itertuples())))
        limits=[axis.get_xlim() for axis in axes[1]]
        for axis in axes.flat: axis.set_xlim(min(a for a,b in limits),max(b for a,b in limits))
        ylimits=[axis.get_ylim() for axis in axes[0]]
        for axis in axes[0]: axis.set_ylim(min(a for a,b in ylimits),max(b for a,b in ylimits))
        handles=[Line2D([],[],marker='^' if kind=='B' else 'v',ls='',color='#444444',label='Valid returned location'),Line2D([],[],marker='x',ls='',color='#444444',label='Invalid returned location')]
        fig.legend(handles=handles,loc='lower center',ncol=2,fontsize=8,frameon=False)
        fig.suptitle(f'{dataset} · {kind}-point {example} example across conditions',fontweight='semibold')
        fig.tight_layout(rect=(0,.04,1,.94))
        stem=f'F01_{dataset}_{kind}_{example}_waveforms'; axis_text='x: time relative to ECG R (ms); y: filtered ICG, beat standardized (a.u.)'
        save(fig,dataset,kind,stem,ledger,axis_text)
        rule={'close':'Smallest complete-beat range among structurally eligible beats with no inherited C timing warning and SNR proxy at or above the participant median; relative proximity, not validated clean agreement.',
              'ambiguous':'Largest complete-beat range, selected without a reference-error outcome.',
              'failed':'Largest number of missed or invalid detections.'}[example]
        report(dataset,kind,stem,'How does actual waveform and candidate geometry differ between conditions?',f'{kind}_master_beats.csv; {kind}_detection_status.csv; original MAT signal files','One participant; two recordings; one selected beat per condition.',md(pd.DataFrame(details)),
               rule+' The other-condition beat is closest to its recording median RR. Source signals are kept in their native polarity and standardized only for display.',
               'Different recordings are paired by participant, not by heartbeat. Shaded windows belong to the named method; other methods retain their own search windows in the CSV. Visual differences alone do not establish mechanism. Large-range or failed examples can be dominated by upstream C warnings or unusually long R-defined regions (possible missed R peaks), not isolated B/X morphology ambiguity; these flags and RR are tabulated explicitly.',
               'Use as an illustrative mechanism check alongside participant-level failure and morphology associations.',registry,axis_text)


def morphology_figures(dataset,kind,tables,settings,registry,ledger,method):
    effects,summaries,associations=morphology_effects(tables,kind,settings['conditions'],method)
    morph=tables[f'{kind}_morphology_features.csv']
    status=tables[f'{kind}_detection_status.csv']
    context=tables[f'{kind}_participant_summary.csv'].query('algorithm_id == @method')
    rr=morph.groupby(['subject_id','condition']).rr_interval_ms.median().rename('median_rr_ms').reset_index()
    rr_pairs=paired(rr,settings['conditions'],'median_rr_ms')
    rr_est,rr_lo,rr_hi=median_ci(rr_pairs.change)
    rr_text=f'RR context: median paired RR change {rr_est:.2f} ms (95% participant bootstrap {rr_lo:.2f} to {rr_hi:.2f} ms). RR shortening is physiology context, not evidence of detection failure.\n\n'+md(rr_pairs)
    counts=[]
    for f in FEATURES:
        cat=f+'_category'
        selected=status.loc[status.algorithm_id.eq(method),KEYS+['attempted','valid','missed','invalid']]
        rows=morph.merge(selected,on=KEYS,validate='one_to_one')
        c=rows.groupby(['condition',cat]).agg(beats=('beat_id','size'),participants=('subject_id','nunique'),attempted=('attempted','sum'),valid=('valid','sum'),missed=('missed','sum'),invalid=('invalid','sum'),common_valid=('range_ms','count')).reset_index().rename(columns={cat:'category'})
        c.insert(0,'feature',f); counts.append(c)
    counts=pd.concat(counts,ignore_index=True)
    for code,outcome,ylabel in [('F06A','failure_pct','Detection failure rate (%)'),('F06B','range_ms','Median within-beat algorithm timing range (ms)')]:
        fig,axes=plt.subplots(2,3,figsize=(11.5,7.2))
        plotted=[]
        for ax,(f,feature_label) in zip(axes.flat,FEATURES.items()):
            for j,condition in enumerate(settings['conditions']):
                for x,cat in enumerate(['low','high']):
                    v=summaries.loc[summaries.feature.eq(f)&summaries.outcome.eq(outcome)&summaries.condition.eq(condition)&summaries.category.eq(cat),'value']
                    est,lo,hi=median_ci(v)
                    if np.isfinite(est):
                        points=ax.errorbar(x+(j-.5)*.12,est,yerr=[[max(0,est-lo)],[max(0,hi-est)]],fmt=['o','s'][j],color=COLOURS[j],capsize=3,ms=4)
                        points.lines[0].set_gid(f'morphology/{outcome}/{method}/{f}/{condition}/{cat}')
                    plotted.append(dict(feature=f,condition=condition,category=cat,participants=int(v.notna().sum()),estimate=est,CI_low=lo,CI_high=hi))
            ax.set_xticks([0,1],['Low','High']); ax.set_xlim(-.3,1.3)
            ax.set(xlabel='Within-participant feature category',ylabel=ylabel,title=feature_label)
        fig.suptitle(f'{dataset} · {kind}-point '+(method+' '+DISPLAY_NAMES[method]+' failure by morphology' if code=='F06A' else 'disagreement by morphology'),fontweight='semibold')
        fig.legend(handles=[Line2D([],[],marker=['o','s'][j],color=COLOURS[j],ls='',label=settings['labels'][j]) for j in range(2)],loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.90))
        stem=f'{code}_{dataset}_{kind}_morphology_'+('failure' if outcome=='failure_pct' else 'disagreement')
        axes_text='x: within-participant morphology category; y: '+ylabel+'; points: medians across participant summaries; intervals: participant bootstrap 95% CI'
        save(fig,dataset,kind,stem,ledger,axes_text)
        e=effects.loc[effects.outcome.eq(outcome)]
        a=associations.loc[associations.outcome.eq(outcome)]
        association_summary=[]
        for f,g in a.groupby('feature'):
            fv,fl,fh=median_ci(g.feature_change); ov,ol,oh=median_ci(g.outcome_change)
            association_summary.append(dict(feature=f,participants=g.subject_id.nunique(),median_feature_change=fv,feature_CI_low=fl,feature_CI_high=fh,median_outcome_change=ov,outcome_CI_low=ol,outcome_CI_high=oh,descriptive_Spearman_rho=g.spearman_rho_descriptive.iloc[0]))
        finite=e.loc[e.paired_participants.ge(3)&((e.ci_low>0)|(e.ci_high<0))]
        supported='; '.join(f'{r.feature}/{r.condition}: {r.high_minus_low:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]' for r in finite.itertuples()) or 'No category contrast has a bootstrap interval wholly away from zero.'
        report(dataset,kind,stem,'Which measurable waveform characteristics are associated with '+('failure?' if outcome=='failure_pct' else 'definition disagreement?'),f'{kind}_morphology_features.csv; {kind}_detection_status.csv',md(counts),
               md(pd.DataFrame(plotted))+'\n\nPaired within-participant high-minus-low effects:\n\n'+md(e)+'\n\nCondition → morphology → outcome paired-change summary (feature units in docs/phase2_methodology.md):\n\n'+md(pd.DataFrame(association_summary))+'\n\n'+rr_text,
               supported+' These are exploratory, unadjusted associations. For failure the displayed method is '+method+'; its selection rule is documented in F05A.',
               'Both successful and failed eligible beats have method-independent region features. A common-valid range is undefined when any method fails; its missingness can be informative. Categories use the participant median across both conditions, with ties assigned low. Sparse/tied categories can be unavailable. The smooth/residual ratio is a filtering-dependent quality proxy, not calibrated SNR. Extrema counts are mathematically related to the availability of a local-minimum candidate, especially B1; this association is partly definitional, not an independent physiological mechanism. Curvature and slope may be redundant; shared RR and window length are possible confounders.',
               'Treat consistent participant effects as candidate explanations requiring independent replication; no causation, prediction model or universal physiological cutoff is claimed.',registry,axes_text)
    return effects,associations


def write_dataset_audit(dataset,settings,inventory,registry,findings):
    inv=[r for r in inventory if r['dataset_id'] in settings['ids']]
    rows=pd.DataFrame([{k:r[k] for k in ['dataset_id','condition','subject_id','recording_id','sampling_rate_hz','samples','duration_s','beats','finite_signals']} for r in inv])
    pivot=rows.groupby(['subject_id','condition']).size().unstack().reindex(columns=settings['conditions'])
    annotation=pd.DataFrame([{'recording_id':r['recording_id'],**r['annotation_events']} for r in inv]).fillna(0)
    text=f'# {dataset} Phase 2 dataset and scientific audit\n\n'
    text+='## Cohort and pairing\n\n'+md(rows.groupby('condition').agg(participants=('subject_id','nunique'),recordings=('recording_id','size'),beats=('beats','sum'),sampling_rate_hz=('sampling_rate_hz','first')).reset_index(),0)
    text+=f'\n\nMatched participant IDs: {len(pivot.dropna())}. Pairing is by parsed source subject ID; one recording per condition is verified. IITB CVD is excluded. No separately recorded beats are called paired heartbeats.\n\nSource metadata: [{dataset} documentation]({settings["source"]}).\n\n'
    text+='## Acquisition, upstream landmarks and annotations\n\n'
    text+=('IITB uses columns 3 (documented −dZ/dt) and 4 (ECG), 500 Hz. Native positive systolic convention is preserved; no sign inversion or additional differentiation is applied. Column 5 is undocumented and unused. Doppler SV is not a B/X timing reference. Full local recordings yield more detected cycles than the selected Doppler cycle subset in the dataset documentation.\n\n' if dataset=='IITB' else 'Local raw-folder MAT files contain synchronized ECG/ICG at 250 Hz according to samplFreq. The public description states acquisition at 2,000 Hz; local files therefore must not be labelled native 2,000-Hz acquisitions. Expert B/C/X/O annotations exist in 46 of 48 recordings, with noisy CW subsets missing. Annotation index origin/timebase and selected clean-segment sampling constrain their use; no accuracy is calculated here.\n\n')
    text+='R peaks are recomputed using the frozen offline Pan–Tompkins implementation. C is the frozen global maximum in R to R+300 ms, flagged outside 60–200 ms. Structural region eligibility is retained exactly; flagged C beats are explicitly visible in master and status tables. Every regenerated R boundary, C sample, candidate status, QC status and timestamp is checked against the accepted source lineage.\n\n'
    text+='## Frozen catalogue\n\n'+md(registry[['algorithm_id','candidate_name','parameters']])
    text+='\n\nX10 remains the provisional Weissler HR formula dependent on B8. Required-dependency failures are explicitly identified under failure_stage, separate from empty regions and missing local features. No detector parameters were changed.\n\n'
    text+='## Recording inventory\n\n'+md(rows)+'\n\n## Annotation availability (counts only; not used as timing truth)\n\n'+md(annotation,0)
    for kind,items in findings.items():
        text+=f'\n\n## {kind}: paired condition changes\n\nFailure (percentage points):\n\n'+md(items['failure'])+'\n\nAcross-method median range (ms):\n\n'+md(items['range'])+'\n\nDefinition-wise differences (ms):\n\n'+md(items['peer'])
    path=RESULTS/'reports'/dataset/'DATASET_AUDIT.md'; path.write_text(text+'\n',encoding='utf-8')


def consolidation(all_tables,findings,inventory,registry):
    out=['# P1 consolidated RQ1 evidence','## 11.1 Dataset and reference audit']
    rows=[]
    for dataset,kinds in all_tables.items():
        for condition,g in kinds['B']['B_master_beats.csv'].groupby('condition'):
            rows.append(dict(dataset=dataset,condition=condition,participants=g.subject_id.nunique(),recordings=g.recording_id.nunique(),beats=len(g),sampling_Hz=g.sampling_frequency_hz.iloc[0]))
    out += [md(pd.DataFrame(rows),0),'HeartCycle is preserved byte-for-byte: 17 participants, 212 recordings, 1,706 beats; ECHO AVO/AVC references. IITB has no validated B/X timing references in the available files. ReBeat expert morphology annotations are available but deliberately not used for main accuracy claims; local raw-folder sampling is 250 Hz.',
            'Metadata sources: [IITB](https://www.ee.iitb.ac.in/~spilab/databases/IITB-ICG-Database/IITB-ICG-Database-Description_2025jul08.pdf), [ReBeat](https://zenodo.org/records/4725433).',
            '## 11.2 Fixed B/X algorithms',md(registry[['algorithm_id','candidate_name']]),'No definitions or thresholds were tuned. X10 remains a provisional HR formula requiring B8. Source-channel polarity and upstream C timing flags limit physiological interpretation.',
            '## 11.3 HeartCycle reference-supported findings']
    hc_morphology={}
    for kind in ('B','X'):
        base=RESULTS/'tables'/kind
        p=pd.read_csv(base/f'{kind}_participant_summary.csv')
        e=pd.read_csv(base/f'{kind}_reference_errors.csv')
        stats=p.groupby('algorithm_id').median_absolute_error_ms.agg(median_ms='median',IQR_ms=lambda x:x.quantile(.75)-x.quantile(.25)).reset_index()
        out += [f'{kind}: {e.beat_id.nunique()} strict common reference beats from {e.subject_id.nunique()} participants. Values below are recomputed read-only from verified tables.',md(stats)]
        d=pd.read_csv(base/f'{kind}_detection_status.csv',low_memory=False)
        failure=d.loc[d.attempted].groupby('algorithm_id').agg(attempts=('beat_id','size'),missed=('missed','sum'),invalid=('invalid','sum')).reset_index()
        failure['failure_pct']=100*(failure.missed+failure.invalid)/failure.attempts
        out+=[md(failure)]
        disagreement=p.drop_duplicates('subject_id')[['subject_id','disagreement_beats','participant_median_range_ms','participant_p95_range_ms']]
        out+=['Participant-level within-beat disagreement (not reference error):',md(disagreement)]
        h=pd.read_csv(base/f'{kind}_morphology_features.csv',low_memory=False)
        h=h.loc[h.strict_common_reference]
        hc_morphology[kind]={}
        for feature in FEATURES:
            cat={'competing':'competing_features_category','variability':'beat_to_beat_variability_category'}.get(feature,feature+'_category')
            cats=['consistent','variable'] if feature=='variability' else ['low','high']
            values=h.loc[h[cat].isin(cats)].groupby(['subject_id','algorithm_id',cat]).absolute_error_ms.median().unstack().reindex(columns=cats).dropna()
            difference=(values[cats[1]]-values[cats[0]]).groupby('algorithm_id').median()
            hc_morphology[kind][feature]=f'{kind}: median paired category-error shifts across definitions {difference.min():.2f} to {difference.max():.2f} ms; {(difference>0).sum()} positive/{(difference<0).sum()} negative definitions (descriptive)'
    out+=['ECHO is independently acquired and aligned through ECG R matching; physiological valve-event versus morphological-landmark offsets remain possible. HeartCycle findings support timing correspondence only under these reference assumptions. Existing HeartCycle reports are not treated as independently validated ground truth when table-derived counts disagree.']
    for index,dataset,title in [('11.4','IITB','exercise-condition'),('11.5','ReBeat','cognitive-condition')]:
        out += [f'## {index} {dataset} {title} findings']
        for kind in ('B','X'):
            item=findings[dataset][kind]
            out += [f'### {kind} failure change (percentage points)',md(item['failure']),f'### {kind} same-beat disagreement change (ms)',md(item['range']),f'### {kind} definition-wise change (ms)',md(item['peer'])]
            t=all_tables[dataset][kind]; m=t[f'{kind}_master_beats.csv']; p=t[f'{kind}_participant_summary.csv']
            retention=p.drop_duplicates(['subject_id','condition']).groupby('condition').agg(common_valid_beats=('common_valid_beats','sum'),excluded=('excluded_from_common_beats','sum')).reset_index()
            out+=['Strict common-valid retention:',md(retention,0)]
            rr=m.groupby(['subject_id','condition']).rr_interval_ms.median().rename('rr').reset_index()
            rc=paired(rr,json.loads((ROOT/'configs/phase2.json').read_text())['datasets'][dataset]['conditions'],'rr')
            value,lo,hi=median_ci(rc.change)
            out+=[f'Paired RR change: median {value:.2f} ms, 95% participant bootstrap [{lo:.2f}, {hi:.2f}]. This can alter search-region duration and range without implying error.']
            effects=item['morphology']; supported=effects.loc[(effects.ci_low>0)|(effects.ci_high<0)]
            out+=['Exploratory morphology contrasts with intervals excluding zero (not multiplicity-adjusted):',md(supported)]
            correlations=item['associations'].groupby(['feature','outcome']).agg(participants=('subject_id','nunique'),median_feature_change=('feature_change','median'),median_outcome_change=('outcome_change','median'),descriptive_rho=('spearman_rho_descriptive','first')).reset_index()
            out+=['Condition → morphology → outcome (paired participant changes; descriptive correlations):',md(correlations)]
    out+=['## 11.6 Cross-dataset morphology and failure-mode synthesis']
    synth=[]
    for feature in FEATURES:
        row={'feature':feature,'HeartCycle reference-error association':hc_morphology['B'][feature]+'; '+hc_morphology['X'][feature], 'HeartCycle failure association':'Not established by preserved F06C; do not substitute error association for failure evidence.'}
        for dataset in ['IITB','ReBeat']:
            entries=[]
            for kind in ['B','X']:
                e=findings[dataset][kind]['morphology']
                e=e.loc[e.feature.eq(feature)]
                for r in e.itertuples():
                    entries.append(f'{kind} {r.outcome} {r.condition}: Δ {r.high_minus_low:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}], n={r.paired_participants}')
            row[dataset]='; '.join(entries)
        synth.append(row)
    out += [md(pd.DataFrame(synth)),'Contrasts are high-minus-low within-participant differences. Intervals are exploratory participant bootstraps without multiplicity correction. Failure examples use the transparently selected method per point type. Similar interval direction is a descriptive replication signal, not a cross-dataset pooled effect. Native amplitude scales, preprocessing and reference-local versus search-region features differ.',
            '## 11.7 Implications for reliable fiducial localisation','HeartCycle reference correspondence, algorithm execution reliability and method disagreement answer different questions. Low failure can coexist with a systematically different landmark. Use paired participant directions and retained-beat denominators before claiming condition sensitivity. Upstream C flags, required B8 dependency and derivative/search-window failures require separate interpretation. Extrema availability is mathematically coupled to local-minimum rules such as B1; that association is not independent evidence of a physiological mechanism. Sharpness/slope can be redundant. No universal winner is selected.',
            '## 11.8 Implications for PEP/LVET','With a fixed Q/R anchor, later B timing increases a B-based pre-ejection interval. LVET = X−B increases with later X and decreases with later B. Correlated B/X shifts may cancel or compound. These qualitative consequences do not constitute biomarker uncertainty propagation or clinical validation.',
            '## 11.9 Limitations and decision gate','This package supports reference-supported timing correspondence in HeartCycle plus descriptive condition-dependent reliability in IITB/ReBeat. It does not establish clinical correctness across conditions. Generalization needs calibrated signal provenance, resolved ReBeat annotation coordinates and selection bias, independent annotated R/C landmarks in difficult recordings, and matched physiological timing references. All beat summaries respect participant clustering; zero common-valid beats and empty morphology strata remain missing. No new clinical tolerance or causal model was introduced.']
    (RESULTS/'reports/P1_CONSOLIDATED_RQ1.md').write_text('\n\n'.join(out)+'\n',encoding='utf-8')


def build_outputs(bundle,before):
    master,status,features,inventory,signals,registry,fingerprints=bundle
    config=json.loads((ROOT/'configs/phase2.json').read_text())
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':100})
    # Back up only previous Phase 2 deliverables before an authorized regeneration.
    paths=[p for family in ['tables','figures','reports'] for ds in config['datasets'] for p in (RESULTS/family/ds).rglob('*') if p.is_file()]
    if paths:
        backup=ROOT/'work/phase2/previous_phase2.zip'
        with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as archive:
            for p in paths: archive.write(p,p.relative_to(ROOT))
    all_tables={}; findings={}; ledger=[]
    for dataset,settings in config['datasets'].items():
        all_tables[dataset]={}; findings[dataset]={}
        for kind in ['B','X']:
            tables=dataset_tables(master,status,features,settings,kind)
            all_tables[dataset][kind]=tables
            directory=RESULTS/'tables'/dataset/kind; directory.mkdir(parents=True,exist_ok=True)
            for name,table in tables.items(): table.to_csv(directory/name,index=False)
            selected,_,fc,rc,pc=condition_figures(dataset,kind,tables,settings,registry,ledger)
            waveform_figures(dataset,kind,tables,settings,registry,signals,ledger,selected)
            effects,associations=morphology_figures(dataset,kind,tables,settings,registry,ledger,selected)
            findings[dataset][kind]={'selected_method':selected,'failure':fc,'range':rc,'peer':pc,'morphology':effects,'associations':associations}
            print(f'Generated {dataset}/{kind} tables, figures and reports',flush=True)
        write_dataset_audit(dataset,settings,inventory,registry,findings[dataset])
    consolidation(all_tables,findings,inventory,registry)
    style=json.loads((ROOT/'configs/figure_style.yaml').read_text())
    legend=registry[['algorithm_id','candidate_name','candidate_type']].copy()
    legend['display_name']=legend.algorithm_id.map(DISPLAY_NAMES)
    legend['colour']=legend.candidate_name.map(style['algorithm_colours'])
    legend['marker']=legend.candidate_name.map(style['algorithm_markers'])
    legend_text='# Phase 2 legend mapping\n\nAlgorithm colours/markers retain configs/figure_style.yaml. Phase 2 condition and participant palettes are configured in configs/phase2.json without changing HeartCycle styling.\n\n'+md(legend)
    for ds in all_tables:
        participants=sorted(all_tables[ds]['B']['B_master_beats.csv'].subject_id.unique())
        legend_text+='\n\n## '+ds+' paired-plot participant key\n\n'+md(pd.DataFrame({'participant':participants,'colour':PARTICIPANT_PALETTE[:len(participants)]}))
    legend_text+='\n\nFirst condition: '+COLOURS[0]+' (circle); second condition: '+COLOURS[1]+' (square). In paired F05 both conditions use circular markers and participant colours. Invalid waveform candidates use x; valid B uses upward triangle, valid X downward triangle. Method-labelled timing rows resolve colour/position overlap. F04 points are individual participants distinguished by condition, not an identity-tracking plot. No reference markers or signed-error colours are used in Phase 2.\n'
    (RESULTS/'reports/PHASE2_LEGEND_MAPPING.md').write_text(legend_text,encoding='utf-8')
    assert before==heartcycle_hashes()
    audit={'heartcycle_unchanged':True,'heartcycle_sha256':before,'input_sha256':fingerprints,'raw_replay':'Every R boundary, C sample, detection/QC status and candidate timestamp matches accepted source lineage',
           'recordings':inventory,'figures':ledger,'participant_colours':{ds:dict(zip(sorted(all_tables[ds]['B']['B_master_beats.csv'].subject_id.unique()),PARTICIPANT_PALETTE)) for ds in all_tables}}
    (ROOT/'work/phase2/execution_audit.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
