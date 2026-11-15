#!/usr/bin/env python3
"""RQ1 CPU-only analysis from blinded ChatGPT JSON results and private manifest.

Usage: python analyze_rq1.py --manifest manifest.json --main chatgpt_main.zip --repeat repeat.zip --out analysis
No GPU, no image decoding. Dep: pandas numpy scipy matplotlib.
"""
import argparse
import json
import zipfile
import hashlib
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCORE_COLS=['consistency_score','instruction_score','quality_score','knowledge_score']
ALL_COLS=['Q_100']+SCORE_COLS
RNG_SEED=20261026


def load_responses(zip_path, prefix):
    results=[]
    with zipfile.ZipFile(zip_path) as z:
        files=[name for name in z.namelist() if re.search(rf'(?:^|/){re.escape(prefix)}_b\d{{3}}\.json$',name) and 'response_template' not in name]
        if not files:
            return pd.DataFrame(),[]
        for name in sorted(files):
            val=json.loads(z.read(name))
            assert isinstance(val,dict) and 'results' in val and isinstance(val['results'],list),(name,'malformed')
            assert val.get('batch_id')==Path(name).stem,(name,val.get('batch_id'))
            for item in val['results']:
                results.append({'file':name,'batch_id':val['batch_id'],**item})
    df=pd.DataFrame(results)
    assert not df.duplicated('blind_id').any(),'Repeated blind ids in response ZIP'
    return df,files


def compute_q(r):
    vals=[r[c] for c in SCORE_COLS if pd.notna(r[c])]
    return float(np.mean([(float(v)-1)*25 for v in vals])) if vals else np.nan


def match_data(manifest, main):
    m=pd.DataFrame(manifest['cases'])
    assert len(m)==len(m.blind_id.unique()),'Manifest blind ids not unique'
    expected=m[m.phase=='main'].copy()
    assert len(expected)==1760 and len(main)==1760, (len(expected),len(main))
    assert set(main.blind_id)==set(expected.blind_id),'Response IDs mismatch manifest'
    assert set(main.columns).issuperset({'blind_id','batch_id','status',*SCORE_COLS,'rationale'})
    d=expected.merge(main,on=['blind_id','batch_id'],how='left',validate='one_to_one',indicator=True)
    assert (d['_merge']=='both').all(),'Unmatched batch ID'
    d.drop(columns=['_merge'],inplace=True)
    for bid,g in d.groupby('batch_id'):
        assert len(g) in (4,6), (bid,len(g))
        assert len(g.example_uid.unique())==1,bid
    assert len(d.batch_id.unique())==380
    assert set(d.status.unique()) <= {'scored','unreadable'},d.status.unique()
    for c in SCORE_COLS:
        bad=d.loc[d.status=='scored',c].dropna().apply(lambda v:not (float(v).is_integer() and 1<=v<=5))
        assert not bad.any(),(c,'invalid scores')
    assert (d.loc[d.status=='scored',['consistency_score','instruction_score','quality_score']].notna().all(axis=1)).all()
    assert d.loc[d.status=='unreadable',SCORE_COLS].isna().all(axis=None)
    assert d.loc[(d.status=='scored') & d.knowledge_applicable, 'knowledge_score'].notna().all()
    assert d.loc[(d.status=='scored') & ~d.knowledge_applicable,'knowledge_score'].isna().all()
    d['Q_100']=d.apply(compute_q,axis=1)
    d['split']=np.where(d.seed==42,'k_test_seed42','k_small_extra_seed')
    assert not d.duplicated(['example_uid','model','condition','seed']).any()
    return d


def descriptive(df, groupby):
    out=df.groupby(groupby,dropna=False).agg(n_expected=('blind_id','size'),n_scored=('Q_100','count'),
      Q_100=('Q_100','mean'),consistency=('consistency_score','mean'),instruction=('instruction_score','mean'),
      quality=('quality_score','mean'),knowledge=('knowledge_score','mean'),knowledge_n=('knowledge_score','count'))
    return out.reset_index()


def pair_frame(data, lhs_model,lhs_condition,rhs_model,rhs_condition, experiment):
    if experiment=='k_test':
        d=data[(data.seed==42)&(data.condition.isin(['base','thinking']))]
    elif experiment=='k_small':
        uids=set(data.loc[data.condition=='correction','example_uid'])
        assert len(uids)==40
        d=data[data.example_uid.isin(uids)]
    else: raise ValueError(experiment)
    l=d[(d.model==lhs_model)&(d.condition==lhs_condition)].copy()
    r=d[(d.model==rhs_model)&(d.condition==rhs_condition)].copy()
    keys=['example_uid','seed']
    common=l.merge(r,on=keys,how='inner',suffixes=('_lhs','_rhs'),validate='one_to_one')
    full_n=300 if experiment=='k_test' else 120
    assert len(common)==full_n,(lhs_model,lhs_condition,rhs_model,rhs_condition,len(common),full_n)
    return common


def paired_numbers(common,metric, multi,seed=RNG_SEED,n_boot=12000,n_perm=50000):
    a=f'{metric}_lhs';b=f'{metric}_rhs'
    pairs=common.dropna(subset=[a,b]).copy()
    pairs['delta']=pairs[a]-pairs[b]
    counts=pairs.groupby('example_uid').size()
    arr=pairs.groupby('example_uid')['delta'].mean().to_numpy(dtype=float)
    # Same sample UID across all seeds is one independent statistical unit.
    # A paired test uses average seed-wise differences per UID, not each seed as independent.
    n=len(arr)
    rng=np.random.default_rng(seed)
    mean=float(arr.mean()) if n else np.nan
    if n:
        boot=np.empty(n_boot)
        for i in range(0,n_boot,500):
            size=min(500,n_boot-i)
            ids=rng.integers(0,n,(size,n))
            boot[i:i+size]=arr[ids].mean(axis=1)
        ci=np.percentile(boot,[2.5,97.5])
        if np.allclose(arr,0): p=1.0
        else:
            extreme=0
            for i in range(0,n_perm,500):
                s=min(500,n_perm-i)
                signs=rng.choice([-1,1],size=(s,n))
                simulated=(signs*arr).mean(axis=1)
                extreme+=np.count_nonzero(np.abs(simulated)>=abs(mean)-1e-12)
            p=(extreme+1)/(n_perm+1)
    else: ci=(np.nan,np.nan);p=np.nan
    return {'n_pairs':len(pairs),'n_examples':n,'missing_pairs':(300 if not multi else 120)-len(pairs),
      'avg_seeds_per_example':float(counts.mean()) if n else np.nan,
      'mean_delta':mean,'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
      'p_signflip':p,'positive_examples':int((arr>0).sum()),'negative_examples':int((arr<0).sum()),
      'ties_examples':int((arr==0).sum())}


def holm(pvals):
    p=np.array(pvals,dtype=float)
    mask=np.isfinite(p)
    ids=np.flatnonzero(mask)[np.argsort(p[mask])]
    adj=np.full(len(p),np.nan)
    prev=0
    for rank, idx in enumerate(ids):
        prev=max(prev,p[idx]*(len(ids)-rank))
        adj[idx]=min(1.0,prev)
    return adj


def make_plots(df, desc, pairs, by_cat, by_seed, out):
    # Q for ktest (baseline vs thinking)
    a=desc[(desc.seed==42)&(desc.condition.isin(['base','thinking']))].copy()
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    names=['Step1X Base','Step1X Thinking','UniReason Base','UniReason Thinking']
    vals=[float(a.loc[(a.model==m)&(a.condition==c),'Q_100'].iloc[0]) for m,c in [('step1x','base'),('step1x','thinking'),('unireason','base'),('unireason','thinking')]]
    bars=ax.bar(names,vals)
    ax.bar_label(bars,fmt='%.1f',padding=3)
    ax.set_ylim(0,100);ax.set_ylabel('Mean applicable-normalized score Q (0–100)');ax.set_title('K-test (300 examples, seed 42): mean score by model and condition')
    ax.tick_params(axis='x',rotation=12)
    fig.tight_layout();fig.savefig(out/'fig_ktest_q.png',dpi=160);plt.close(fig)
    # Paired primary contrasts with CIs.
    d=pairs[(pairs.metric=='Q_100')].copy()
    fig,ax=plt.subplots(figsize=(9.5,5.4))
    y=np.arange(len(d))[::-1]
    errs=np.vstack([d.mean_delta-d.ci95_low,d.ci95_high-d.mean_delta])
    ax.errorbar(d.mean_delta,y,xerr=errs,fmt='o',capsize=4)
    ax.axvline(0,linestyle='--')
    ax.set_yticks(y,d.contrast.str.replace('Step1X','Step1X',regex=False))
    ax.set_xlabel('Mean paired difference in Q (percentage points; lhs – rhs)')
    ax.set_title('Paired RQ1 comparisons, 95% example-level bootstrap intervals')
    fig.tight_layout();fig.savefig(out/'fig_paired_effects.png',dpi=160);plt.close(fig)
    # Metrics descriptive per condition on ktest, grouped four conditions.
    d=df[(df.seed==42)&df.condition.isin(['base','thinking'])].copy()
    ag=d.groupby(['model','condition'])[SCORE_COLS].mean()
    index=[('step1x','base'),('step1x','thinking'),('unireason','base'),('unireason','thinking')]
    fig,ax=plt.subplots(figsize=(9,4.5))
    n=len(index);xx=np.arange(n); width=.2
    for j,metric in enumerate(SCORE_COLS):
        values=[ag.loc[idx,metric] for idx in index]
        ax.bar(xx+(j-1.5)*width,values,width,label=metric.replace('_score','').title())
    ax.set_ylim(0,5);ax.set_xticks(xx,['Step1X Base','Step1X Thinking','UniReason Base','UniReason Thinking'],rotation=10)
    ax.set_ylabel('Mean rubric score (1–5)');ax.set_title('K-test by rubric dimension (knowledge scored only where applicable)');ax.legend(ncol=2)
    fig.tight_layout();fig.savefig(out/'fig_ktest_rubrics.png',dpi=160);plt.close(fig)
    # per category thinking gains (paired) n>=?
    pivot=by_cat[(by_cat.comparison=='Thinking − Base')&(by_cat.metric=='Q_100')].copy()
    if len(pivot):
        fig,ax=plt.subplots(figsize=(10,7))
        categories=sorted(pivot.category.unique());u=np.arange(len(categories));ww=.36
        for j,model in enumerate(['step1x','unireason']):
            vals=[float(pivot.loc[(pivot.category==cat)&(pivot.model==model),'mean_delta'].iloc[0]) if len(pivot.loc[(pivot.category==cat)&(pivot.model==model)]) else np.nan for cat in categories]
            ax.barh(u+(j-.5)*ww,vals,height=ww,label=model)
        ax.set_yticks(u,categories);ax.axvline(0,linestyle='--');ax.set_xlabel('Thinking – Base mean Q difference (seed 42)');ax.set_title('Per-category thinking effect (descriptive)');ax.legend()
        fig.tight_layout();fig.savefig(out/'fig_category_thinking.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--main',type=Path,required=True);ap.add_argument('--repeat',type=Path);ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text()); responses,files=load_responses(a.main,'main')
    data=match_data(manifest,responses)
    data.to_csv(a.out/'individual_scores_private.csv',index=False)
    desc_all=descriptive(data,['model','condition'])
    desc_all.round(5).to_csv(a.out/'descriptive_all_available.csv',index=False)
    desc_by_seed=descriptive(data,['model','condition','seed'])
    desc_by_seed.round(5).to_csv(a.out/'descriptive_by_seed.csv',index=False)
    desc_ktest=descriptive(data[(data.seed==42)&(data.condition.isin(['base','thinking']))],['model','condition'])
    desc_ktest.round(5).to_csv(a.out/'descriptive_ktest.csv',index=False)
    smallids=set(data.loc[data.condition=='correction','example_uid'])
    desc_ksmall=descriptive(data[data.example_uid.isin(smallids)],['model','condition'])
    desc_ksmall.round(5).to_csv(a.out/'descriptive_ksmall.csv',index=False)
    configs=[
      ('Step1X: Thinking − Base','step1x','thinking','step1x','base','k_test'),
      ('UniReason: Thinking − Base','unireason','thinking','unireason','base','k_test'),
      ('Step1X: Correction − Thinking','step1x','correction','step1x','thinking','k_small'),
      ('UniReason: Correction − Thinking','unireason','correction','unireason','thinking','k_small'),
      ('Step1X − UniReason: Base','step1x','base','unireason','base','k_test'),
      ('Step1X − UniReason: Thinking','step1x','thinking','unireason','thinking','k_test'),
      ('Step1X − UniReason: Correction','step1x','correction','unireason','correction','k_small')]
    pairrows=[];by_category=[];by_seed=[]
    for idx,(name,lm,lc,rm,rc,subset) in enumerate(configs):
        cm=pair_frame(data,lm,lc,rm,rc,subset)
        for metric in ALL_COLS:
            stat=paired_numbers(cm,metric,subset=='k_small',seed=RNG_SEED+100*idx+ALL_COLS.index(metric))
            pairrows.append({'contrast':name,'lhs_model':lm,'lhs_condition':lc,'rhs_model':rm,'rhs_condition':rc,
              'subset':subset,'metric':metric,**stat})
        for cat,cmcat in cm.groupby('category_lhs'):
            for metric in ALL_COLS:
                col1=metric+'_lhs';col2=metric+'_rhs'
                d=cmcat.dropna(subset=[col1,col2]).copy()
                if len(d)==0:continue
                d['delta']=d[col1]-d[col2]
                uidagg=d.groupby('example_uid')['delta'].mean()
                by_category.append({'contrast':name,'comparison':('Thinking − Base' if lc=='thinking' and rc=='base' else 'Correction − Thinking' if lc=='correction' and rc=='thinking' else 'Cross-model'),
                  'model':lm,'category':cat,'metric':metric,'n_pairs':len(d),'n_examples':len(uidagg),'mean_delta':uidagg.mean()})
        if subset=='k_small':
            for seed_,cms in cm.groupby('seed'):
                for metric in ALL_COLS:
                    lhs=metric+'_lhs';rhs=metric+'_rhs'; d=cms.dropna(subset=[lhs,rhs]);
                    by_seed.append({'contrast':name,'seed':seed_,'metric':metric,'n_pairs':len(d),'mean_delta':(d[lhs]-d[rhs]).mean()})
    pairs=pd.DataFrame(pairrows)
    qmask=pairs.metric=='Q_100'
    pairs.loc[qmask,'p_holm_7_primary_Q']=holm(pairs.loc[qmask,'p_signflip'].values)
    pairs.to_csv(a.out/'paired_comparisons_all_metrics.csv',index=False)
    catdf=pd.DataFrame(by_category);catdf.to_csv(a.out/'paired_category_differences.csv',index=False)
    seeddf=pd.DataFrame(by_seed);seeddf.to_csv(a.out/'paired_seed_differences.csv',index=False)
    d=data[data.status=='unreadable'][['blind_id','example_uid','model','condition','seed','category','batch_id','status']]
    d.to_csv(a.out/'unreadable.csv',index=False)
    # identify strongest negative/positive paired cases for human RQ1.3 check
    examples=[]
    for name,lm,lc,rm,rc,subset in configs:
        cm=pair_frame(data,lm,lc,rm,rc,subset)
        cm=cm.dropna(subset=['Q_100_lhs','Q_100_rhs']).copy()
        cm['delta']=cm.Q_100_lhs-cm.Q_100_rhs
        for (uid,cat),g in cm.groupby(['example_uid','category_lhs']):
            examples.append({'contrast':name,'example_uid':uid,'category':cat,'n_seeds':len(g),
                 'mean_delta':g.delta.mean(), 'mean_lhs_Q':g.Q_100_lhs.mean(), 'mean_rhs_Q':g.Q_100_rhs.mean(),
                 'lhs_blind_ids':';'.join(g.blind_id_lhs.tolist()),'rhs_blind_ids':';'.join(g.blind_id_rhs.tolist())})
    pd.DataFrame(examples).to_csv(a.out/'case_selection_for_human_review_PRIVATE.csv',index=False)
    # repeat evidence: zip contains blank templates or scored results?
    repeat_note='No repeat responses archive provided.'
    if a.repeat and a.repeat.exists():
        repeatresp,repeatfiles=load_responses(a.repeat,'repeat')
        repeatcases=manifest['counts']['repeat_ratings']
        if len(repeatresp)==0:
            with zipfile.ZipFile(a.repeat) as z:
                candidate=[n for n in z.namelist() if n.endswith('response_template.json')]
            repeat_note=f'Repeat ZIP contains {len(candidate)} blank response templates and no scored repeat_*.json files. Judge repeatability cannot be computed.'
        else:
            rm=pd.DataFrame([c for c in manifest['cases'] if c['phase']=='repeat'])
            if not (set(repeatresp.blind_id)==set(rm.blind_id)):
                raise ValueError('Repeat response IDs mismatch')
            rr=rm.merge(repeatresp,on=['batch_id','blind_id'],validate='one_to_one')
            left=data.rename(columns={c:c+'_original' for c in SCORE_COLS+['Q_100']})
            joined=rr.merge(left[['blind_id']+[c+'_original' for c in SCORE_COLS+['Q_100']]],left_on='repeat_of',right_on='blind_id',suffixes=('','_original_id'),validate='one_to_one')
            joined['Q_100']=joined.apply(compute_q,axis=1)
            repeat_rows=[]
            for metric in ALL_COLS:
                scores=joined[[metric,metric+'_original']].dropna()
                differences=(scores[metric]-scores[metric+'_original']).abs()
                repeat_rows.append({'metric':metric,'n':len(differences),'mean_abs_diff':differences.mean(), 'p95_abs_diff':differences.quantile(.95), 'exact_agreement':float((differences==0).mean()) if len(differences) else np.nan})
            pd.DataFrame(repeat_rows).to_csv(a.out/'repeatability.csv',index=False)
            repeat_note=f'Repeat ratings present: {len(joined)}, analyzed separately.'
    make_plots(data,desc_by_seed,pairs,catdf,seeddf,a.out)
    # Design/data quality and reusable human report.
    title=['# RQ1 CPU-only ChatGPT evaluation analysis','',
      '> **Internal/private analysis.** Individual case CSVs include model identities and should NOT be sent to a blinded judge.','',
      '## Data validation','',
      f'- Main responses: {len(files)} complete batch JSON files, {len(data)} distinct case IDs.',
      f'- Scored: {(data.status=="scored").sum()}; unreadable: {(data.status=="unreadable").sum()} (omitted from metric means and paired comparisons).',
      f'- Knowledge metric applicable to {data.knowledge_applicable.sum()} items; scored on {data.knowledge_score.count()} items.',
      '- Every main case is matched to its private manifest record; expected batch sizes and score ranges verified.',
      f'- Repeat files: {repeat_note}',
      '',
      '## Definition of aggregate Q','',
      'For each scored image, compute `Q = mean(25 × (score − 1))` over its applicable consistency, instruction, quality and, where applicable, knowledge scores. The resulting Q is on a 0–100 scale. This is an explicitly defined *manual UI aggregate*, NOT a claim of official KRIS API equivalence. Per-metric results are reported separately.','',
      '## K-test (seed 42, 300 original examples)','',
      'This is the primary 300-example Base-vs-Thinking comparison; correction is evaluated only on K-small.', '',
      '| Model | Condition | n scored | Mean Q | Consistency | Instruction | Quality | Knowledge (applicable) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for _,r in desc_ktest.iterrows():
        title.append(f'| {r.model} | {r.condition} | {int(r.n_scored)} | {r.Q_100:.2f} | {r.consistency:.2f} | {r.instruction:.2f} | {r.quality:.2f} | {r.knowledge:.2f} |')
    title.extend(['','## Paired comparisons', '',
      'Mean differences and bootstrap intervals are paired by original example UID. For 3-seed K-small, seed-level paired differences are averaged *within UID* before resampling/testing, so 120 correlated outputs are not treated as 120 independent examples. CI: 12,000 example-level bootstrap resamples; p: 50,000 two-sided paired sign-flip draws, with Holm correction across the seven Q comparisons. p-values depend on exchangeability/symmetry assumptions and are exploratory, especially for K-small.','',
      '| Comparison (LHS − RHS) | Set | n UIDs | matched outputs | ΔQ | 95% CI | p (Holm across 7) |','|---|---|---:|---:|---:|---|---:|'])
    for _,r in pairs[pairs.metric=='Q_100'].iterrows():
        title.append(f'| {r.contrast} | {r.subset} | {r.n_examples} | {r.n_pairs} | {r.mean_delta:+.2f} | [{r.ci95_low:+.2f}, {r.ci95_high:+.2f}] | {r.p_holm_7_primary_Q:.4g} |')
    title.extend(['','## Research interpretation and limitations','',
      '- These are masked, manual ChatGPT UI ratings; prompts group multiple edited images. Blinding helps, but within-batch comparisons may still influence scores.',
      '- Inspect individual rubric dimensions before asserting that reasoning improves instruction compliance or knowledge correctness; Q combines scores with different applicability.',
      '- The K-test sample contains 300 UIDs for Base/Thinking; K-small contains 40 UIDs × 3 seeds for Correction/Thinking; never generalize correction results to the entire K-test without qualification.',
      '- Missing/unreadable outputs are excluded rather than assigned an arbitrary 1. Check sensitivity to missingness and include the unreadable-case table in reporting.',
      '- Category and individual-case tables are descriptive/exploratory, not human-verified failure taxonomy labels. Human RQ1.3 annotation remains outstanding.',
      '- Judge reliability cannot be established from blank repeat templates; obtain `repeat_b001.json` ... `repeat_b010.json` with actual scored responses.',
      '', '## Files','',
      '- `descriptive_ktest.csv`, `descriptive_ksmall.csv`, `descriptive_all_available.csv`, `descriptive_by_seed.csv`',
      '- `paired_comparisons_all_metrics.csv`: the seven planned contrasts for all four rubrics plus Q',
      '- `paired_category_differences.csv`, `paired_seed_differences.csv`, `unreadable.csv`',
      '- `individual_scores_private.csv`, `case_selection_for_human_review_PRIVATE.csv`: model-ID revealed; internal only',
      '- Four figures: K-test aggregate, K-test component rubrics, paired Q effects, per-category thinking effects',
      '- `analyze_rq1.py`: reproducible CPU analysis',
      ''])
    (a.out/'ANALYSIS_REPORT.md').write_text('\n'.join(title))
    summary={'validation': {'main_batches':len(files),'main_ratings':len(data),'scored':int((data.status=='scored').sum()),'unreadable':int((data.status=='unreadable').sum())},
       'repeat_note':repeat_note,'results_Q':pairs[pairs.metric=='Q_100'].to_dict(orient='records')}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print('SUCCESS:',a.out)
    print(desc_ktest[['model','condition','n_scored','Q_100','consistency','instruction','quality','knowledge']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nPRIMARY PAIRED Q:\n',pairs[pairs.metric=='Q_100'][['contrast','n_examples','n_pairs','mean_delta','ci95_low','ci95_high','p_signflip','p_holm_7_primary_Q']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('REPEAT:',repeat_note)

if __name__=='__main__':main()
