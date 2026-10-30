#!/usr/bin/env python3
"""CPU-only RQ1 Astra repeat-score validation and reliability analysis.

Example:
 python rq1_repeat_analysis.py --responses 'responses(1).zip' \
   --manifest manifest.json --previous-main chatgpt_main.zip --out ./rq1_repeat_report

Does not require generated images, torch, or a GPU.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

METRICS = ('consistency_score', 'instruction_score', 'quality_score', 'knowledge_score')

def load_jsons(zip_path):
    result={}
    with ZipFile(zip_path) as z:
        for filename in z.namelist():
            if not filename.endswith('.json'):
                continue
            name=Path(filename).name
            if not (name.startswith('main_b') or name.startswith('repeat_b')):
                continue
            if name in result:
                raise ValueError('Duplicate batch '+name)
            result[name]=json.loads(z.read(filename))
    return result

def q100(row, applicability):
    if row['status'] != 'scored': return float('nan')
    names=METRICS if applicability else METRICS[:3]
    return float(np.mean([(row[k]-1)*25 for k in names]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--responses',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--previous-main',type=Path)
    p.add_argument('--out',type=Path,default=Path('rq1_repeat_report'))
    args=p.parse_args()
    m=json.loads(args.manifest.read_text())
    manifest_cases={c['blind_id']:c for c in m['cases']}
    batches=load_jsons(args.responses)
    by_batch=defaultdict(list)
    for c in m['cases']:by_batch[c['batch_id']].append(c)
    assert len(m['cases'])==1780 and len(manifest_cases)==1780
    assert len(by_batch)==390 and len(batches)==390
    assert set(by_batch)=={p[:-5] for p in batches}
    response_by_id={}
    bad=[]
    for fname,d in batches.items():
        expected=fname[:-5]
        assert d['batch_id']==expected,(fname,d['batch_id'])
        actual={r['blind_id'] for r in d['results']}
        require={c['blind_id'] for c in by_batch[expected]}
        assert len(d['results'])==len(actual)==len(require),(fname,'duplicate or wrong size')
        assert actual==require,(fname,'unexpected/missing IDs')
        for r in d['results']:
            bid=r['blind_id']; c=manifest_cases[bid]
            assert bid not in response_by_id,bid
            response_by_id[bid]=r
            s=r.get('status')
            if s=='scored':
                for k in METRICS[:3]:
                    assert type(r[k]) is int and 1<=r[k]<=5,(fname,bid,k)
                if c['knowledge_applicable']:
                    assert type(r['knowledge_score']) is int and 1<=r['knowledge_score']<=5,(fname,bid)
                else:
                    assert r['knowledge_score'] is None,(fname,bid)
            elif s=='unreadable':
                assert all(r[k] is None for k in METRICS),(fname,bid)
            else:
                bad.append((fname,bid,s))
    assert not bad,bad
    assert set(response_by_id)==set(manifest_cases)
    assert sum(fname.startswith('main_b') for fname in batches)==380
    assert sum(fname.startswith('repeat_b') for fname in batches)==10

    main_unchanged=None
    if args.previous_main:
        earlier=load_jsons(args.previous_main)
        now={k:v for k,v in batches.items() if k.startswith('main_')}
        assert len(earlier)==380 and all(k in earlier and earlier[k]==v for k,v in now.items())
        main_unchanged=True

    repeat=[]
    for c in m['cases']:
        if c['phase']!='repeat':continue
        old=manifest_cases.get(c['repeat_of'])
        assert old and old['phase']=='main'
        assert all(old[k]==c[k] for k in ('edited_sha256','example_uid','model','condition','seed','category','knowledge_applicable'))
        old_r=response_by_id[old['blind_id']]
        new_r=response_by_id[c['blind_id']]
        assert old_r['status']=='scored' and new_r['status']=='scored',('Cannot score repeat',c['blind_id'])
        old_q=q100(old_r,c['knowledge_applicable'])
        new_q=q100(new_r,c['knowledge_applicable'])
        record={
            'repeat_batch':c['batch_id'],'original_blind_id':old['blind_id'],
            'repeat_blind_id':c['blind_id'],'example_uid':c['example_uid'],
            'model':c['model'],'condition':c['condition'],'seed':c['seed'],
            'knowledge_applicable':c['knowledge_applicable'],
            'Q_first':old_q,'Q_repeat':new_q,
            'Q_diff_repeat_minus_first':new_q-old_q,
            'Q_absolute_difference':abs(new_q-old_q),
        }
        for k in METRICS:
            record['first_'+k]=old_r[k]
            record['repeat_'+k]=new_r[k]
            record['diff_'+k]=None if old_r[k] is None else new_r[k]-old_r[k]
        repeat.append(record)
    assert len(repeat)==20

    args.out.mkdir(parents=True,exist_ok=True)
    csvpath=args.out/'repeat_pair_scores_PRIVATE.csv'
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(repeat[0]))
        writer.writeheader();writer.writerows(repeat)
    delta=np.array([r['Q_diff_repeat_minus_first'] for r in repeat],float)
    ad=np.abs(delta)
    rng=np.random.default_rng(20261028)
    bs=np.mean(rng.choice(ad,(20000,len(ad)),replace=True),axis=1)
    stats={
        'main_batches':380,'repeat_batches':10,'total_cases':1780,
        'main_cases':1760,'repeat_cases':20,
        'main_scored':sum(response_by_id[c['blind_id']]['status']=='scored' for c in m['cases'] if c['phase']=='main'),
        'repeat_scored':len(repeat),
        'main_json_matches_previous_upload':main_unchanged,
        'Q100_definition':'mean(25*(rubric score - 1)) over applicable 1-5 rubric fields',
        'Q_mean_first':float(np.mean([r['Q_first'] for r in repeat])),
        'Q_mean_repeat':float(np.mean([r['Q_repeat'] for r in repeat])),
        'Q_mean_signed_difference_repeat_minus_first':float(delta.mean()),
        'Q_MAD':float(ad.mean()),
        'Q_MAD_bootstrap95':list(map(float,np.percentile(bs,[2.5,97.5]))),
        'Q_MedianAD':float(np.median(ad)),
        'Q_95th_percentile_AD_linear':float(np.percentile(ad,95)),
        'Q_maxAD':float(ad.max()),
        'Q_exact':int(sum(ad<1e-8)),
        'Q_exact_pct':float(np.mean(ad<1e-8)*100),
        'selection_by_model_condition':{f'{model}_{cond}':len([r for r in repeat if r['model']==model and r['condition']==cond]) for model,cond in sorted({(r['model'],r['condition']) for r in repeat})},
        'metric_agreement':{},
        'limitations':[
            'Only 20 repeated outputs drawn from 10 source examples; p95 is sample-sensitive.',
            'Repeat selection contains Step1X Thinking and UniReason Correction only, so does not represent every model/condition.',
            'Archive contains no repeat-session model/settings logs, so fresh-chat/model protocol cannot be independently verified from JSON alone.',
            'Within-batch absolute ratings are adapted from KRIS official rubrics, not the official pinned API evaluator.',
            'Image-level repeatability is not an effect-size threshold for group-mean contrasts.'
        ]
    }
    for k in METRICS:
        changes=np.array([r['diff_'+k] for r in repeat if r['diff_'+k] is not None],float)
        stats['metric_agreement'][k]={
            'n':int(len(changes)),
            'exact':int(sum(changes==0)),
            'exact_pct':float(100*np.mean(changes==0)),
            'within_one_point':int(sum(abs(changes)<=1)),
            'MAD_on_1_to_5':float(np.mean(abs(changes))),
            'max_abs_diff':float(np.max(abs(changes)))
        }
    (args.out/'repeatability_summary.json').write_text(json.dumps(stats,indent=2)+'\n')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,5))
        x=np.array([r['Q_first'] for r in repeat]);y=np.array([r['Q_repeat'] for r in repeat])
        ax.scatter(x,y,s=43,alpha=0.75)
        ax.plot([0,100],[0,100],linestyle='--')
        ax.set_xlim(-3,103);ax.set_ylim(-3,103)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('Original aggregate score Q (0–100)')
        ax.set_ylabel('Repeat aggregate score Q (0–100)')
        ax.set_title('RQ1 Astra UI judge repeatability (n=20)')
        fig.tight_layout();fig.savefig(args.out/'repeatability_scatter.png',dpi=170);plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
