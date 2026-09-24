"""Checkpointed independent full-Bayesian reproduction and 2x2 comparison.

Usage: .venv-independent/bin/python -m src.independent.run --stage baseline
       .venv-independent/bin/python -m src.independent.run --stage compare
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar

from src import trials
from src.util import DATA,write_json,utcnow
from src.independent.bayes import OUT,fit,reference_data,serve_draws
from src.independent.scoring import posterior_iid_vectorized,posterior_burst,rules

RATES=[0.,.1,.2,.35,.5,.7,1.]
CAP=2.5
SIMS=16384
CACHE=DATA/'independent_cache'


def sampler_progress():
    """Observability only; leaves the probability model and sampler settings alone."""
    import nutpie
    original=nutpie.sample
    def wrapped(*args,**kwargs):
        previous=kwargs.get('progress_callback')
        def progress(chains):
            if previous is not None:previous(chains)
            print('SAMPLING',[(c.finished_draws,c.total_draws,c.tuning) for c in chains],flush=True)
        kwargs.update(progress_callback=progress,progress_rate=30000)
        return original(*args,**kwargs)
    nutpie.sample=wrapped


def arm_name(cap):return 'ordinary' if cap is None else f'robust_{cap}'.replace('.','p')


def period_prediction(period,cap,bursts=False,seed=4001):
    meta=fit(period,cap)
    meta['encoding']['last_period']=[int(t) for t in meta['encoding']['last_period']]
    arm=arm_name(cap)
    rows=reference_data().query('period == @period').copy()
    stem=OUT/'fits'/f'{arm}_period_{period}'
    cache=CACHE/f'{arm}_period_{period}.npz'
    CACHE.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        z=np.load(cache);pa=z['pa'];pb=z['pb'];p=z['iid'];pub=z['published_iid']
        assert np.array_equal(z['match_ids'],rows.match_id.to_numpy().astype(str))
    else:
        # Materialize the compressed archive once. Passing NpzFile through to
        # serve_draws would decompress the large surface array on every lookup.
        with np.load(stem.with_suffix('.npz')) as archive:
            arrays={name:archive[name] for name in archive.files}
        pa,pb=serve_draws(rows,arrays,meta['encoding'],seed=8000+period)
        bo=rows.best_of.to_numpy()
        fmt=np.array([rules(r.tournament,r.year,r.best_of) for r in rows.itertuples()])
        pub=posterior_iid_vectorized(pa,pb,bo,np.full(len(rows),6),np.full(len(rows),7))
        p=pub.copy()
        changed=(fmt[:,0]!=6)|(fmt[:,1]!=7)
        if changed.any():
            p[changed]=posterior_iid_vectorized(pa[changed],pb[changed],bo[changed],
                                                fmt[changed,0],fmt[changed,1])
        np.savez_compressed(cache,pa=pa,pb=pb,iid=p,published_iid=pub,
                            match_ids=rows.match_id.to_numpy().astype(str))
    rows['iid']=p;rows['published_iid']=pub
    if bursts:
        dest=CACHE/f'{arm}_period_{period}_burst_{SIMS}_{seed}.npy'
        if dest.exists():q=np.load(dest)
        else:
            fmt=np.array([rules(r.tournament,r.year,r.best_of) for r in rows.itertuples()])
            print('SIMULATE',arm,period,'matches',len(rows),'S',SIMS,'seed',seed,flush=True)
            q=posterior_burst(pa,pb,rows.best_of.to_numpy(),fmt[:,0],fmt[:,1],SIMS,seed+period*1000003)
            np.save(dest,q)
            trials.log(stage='independent-full-bayes',kind='conditional-burst-simulation',
                       params={'arm':arm,'period':period,'S':SIMS,'seed':seed,
                               'severity':1.5,'duration':40,'start_uniform':[0,80]})
        assert len(q)==len(rows)
        rows['conditional_burst']=q
    return rows


def season(year,cap=None,bursts=False,seed=4001):
    assert year in [2013,2014]
    periods=range((year-2011)*6,(year-2011)*6+6)
    frame=pd.concat([period_prediction(t,cap,bursts,seed) for t in periods],ignore_index=True)
    frame.to_csv(OUT/f'{arm_name(cap)}_{year}_predictions_{seed}.csv',index=False)
    return frame


def ci(values,rows,unit='tournament',B=3000,seed=17):
    """Resample whole events or both unordered player-event memberships."""
    values=np.asarray(values,float)
    assert len(values)==len(rows)>0 and np.isfinite(values).all()
    rng=np.random.default_rng(seed)
    boot=np.empty(B)
    if unit=='tournament':
        code,unique=pd.factorize(rows.tourney_id)
        sx=np.bincount(code,weights=values);n=np.bincount(code)
        for b in range(B):
            ix=rng.integers(len(unique),size=len(unique))
            boot[b]=sx[ix].sum()/n[ix].sum()
    elif unit=='player_tournament':
        a=(rows.tourney_id+'|'+rows.p1).to_list();b=(rows.tourney_id+'|'+rows.p2).to_list()
        nodes={k:i for i,k in enumerate(sorted(set(a+b)))}
        ia=np.array([nodes[k] for k in a]);ib=np.array([nodes[k] for k in b])
        for j in range(B):
            # Pigeonhole bootstrap: retain both endpoints, not winner-only clusters.
            while True:
                w=rng.poisson(1.,len(nodes));weights=w[ia]*w[ib]
                if weights.sum()>0:break
            boot[j]=np.dot(weights,values)/weights.sum()
    else:raise ValueError(unit)
    lo,hi=np.quantile(boot,[.025,.975])
    return {'mean':float(values.mean()),'lo':float(lo),'hi':float(hi),'n':len(values)}


def score(p,rows):
    p=np.clip(p,1e-10,1-1e-10)
    acc=(p>.5).astype(float)+.5*(p==.5)
    out={}
    for metric,val in [('log_loss',-np.log(p)),('accuracy',acc),('brier',(1-p)**2)]:
        out[metric]={'tournament':ci(val,rows),'player_tournament':ci(val,rows,'player_tournament')}
    # The fixture is stored winner-first. Mirror every forecast to reconstruct
    # an orientation-neutral binary sample for standard calibration summaries.
    q=np.r_[p,1-p];y=np.r_[np.ones(len(p)),np.zeros(len(p))]
    slope=minimize_scalar(lambda s:-np.mean(y*np.log(expit(s*logit(q)))+
                                            (1-y)*np.log(expit(-s*logit(q)))),
                          bounds=(0,5),method='bounded').x
    bins=np.minimum((q*10).astype(int),9)
    ece=0.
    for b in range(10):
        mask=bins==b
        if mask.any():ece+=mask.mean()*abs(q[mask].mean()-y[mask].mean())
    out['calibration']={'slope':float(slope),'ece_10':float(ece),
                        'construction':'winner-first forecasts mirrored to both orientations'}
    return out


def baseline():
    d=season(2014)
    result={'utc':utcnow(),'n':len(d),'published_reference':{'log_loss':.592,'accuracy':.688},
            'ordinary_published_rules':score(d.published_iid,d),
            'ordinary_historical_rules':score(d.iid,d)}
    ll=result['ordinary_published_rules']['log_loss']['tournament']['mean']
    result['within_existing_002_tolerance']=bool(abs(ll-.592)<.02)
    write_json(OUT/'baseline.json',result)
    trials.log(stage='independent-full-bayes',kind='baseline_2014',params={'start':2011,'period_months':2},
               metrics={'log_loss':ll,'n':len(d)})
    print('BASELINE 2014',ll,flush=True)
    return result


def strata(rows):
    order={'R128':0,'R64':1,'R32':2,'R16':3,'QF':4,'SF':5,'F':6,'RR':-1}
    ranks=rows['round'].map(order)
    result=np.full(len(rows),'unknown',dtype=object)
    for tid,inds in rows.groupby('tourney_id').groups.items():
        seen=set()
        for ix in sorted(inds,key=lambda ix: ranks.loc[ix] if pd.notna(ranks.loc[ix]) else -1):
            if rows.loc[ix,'round']=='RR' or pd.isna(ranks.loc[ix]):continue
            a,b=rows.loc[ix,['p1','p2']]
            first=int(a not in seen)+int(b not in seen)
            result[ix]=['both_later','mixed_first_later','both_first'][first]
            seen.update([a,b])
    return result


def fit_diagnostics():
    result={}
    for cap in [None,CAP]:
        arm=arm_name(cap);periods=[]
        for period in range(12,24):
            item=json.loads((OUT/'fits'/f'{arm}_period_{period}.json').read_text())
            d=item['diagnostics']
            periods.append({'period':period,'max_rhat':d['max_rhat'],
                            'min_ess_bulk':d['min_ess_bulk'],
                            'divergences':d['divergences'],'seconds':item['seconds']})
        result[arm]={'max_rhat':max(x['max_rhat'] for x in periods),
                     'min_ess_bulk':min(x['min_ess_bulk'] for x in periods),
                     'divergences':sum(x['divergences'] for x in periods),
                     'periods_rhat_over_1p01':[x['period'] for x in periods if x['max_rhat']>1.01],
                     'periods_rhat_over_1p05':[x['period'] for x in periods if x['max_rhat']>1.05],
                     'periods':periods}
    return result


def compare():
    if not (OUT/'baseline.json').exists():baseline()
    curves={};chosen={};evaluations={};rates={}
    for cap in [None,CAP]:
        arm=arm_name(cap)
        val=season(2013,cap,True)
        curve=[]
        for rate in RATES:
            p=(1-rate)*val.iid.to_numpy()+rate*val.conditional_burst.to_numpy()
            ll=float(-np.log(np.clip(p,1e-10,1)).mean())
            curve.append({'rate':rate,'log_loss':ll})
            trials.log(stage='independent-full-bayes',kind='tune_burst_2013',params={'arm':arm,'rate':rate},metrics={'log_loss':ll})
        rate=min(curve,key=lambda x:x['log_loss'])['rate']
        curves[arm]=curve;rates[arm]=rate
        write_json(OUT/f'{arm}_frozen_forecast.json',{'utc':utcnow(),'tune_year':2013,'rate':rate,'curve':curve})
        d=season(2014,cap,True)
        evaluations[arm]=d
        chosen[arm+'_iid']=d.iid.to_numpy()
        chosen[arm+'_burst']=(1-rate)*d.iid.to_numpy()+rate*d.conditional_burst.to_numpy()
        if cap is None:
            temps=[.75,.9,1.,1.1,1.25,1.5,2.]
            tl=[float(-np.log(expit(logit(np.clip(val.iid,1e-8,1-1e-8))/t)).mean()) for t in temps]
            temp=temps[int(np.argmin(tl))]
            chosen['ordinary_temperature']=expit(logit(np.clip(d.iid,1e-8,1-1e-8))/temp)
            temperature={'chosen':temp,'curve':dict(zip(temps,tl))}
            for t,ll in zip(temps,tl):
                trials.log(stage='independent-full-bayes',kind='tune_temperature_2013',params={'temperature':t},metrics={'log_loss':ll})
    d=evaluations['ordinary']
    assert d.match_id.equals(evaluations[arm_name(CAP)].match_id)
    metrics={k:score(v,d) for k,v in chosen.items()}
    comparisons={}
    labels=strata(d)
    pairs=[('robust_2p5_burst','ordinary_iid'),('robust_2p5_burst','ordinary_burst'),
           ('robust_2p5_iid','ordinary_iid'),('ordinary_burst','ordinary_iid'),
           ('ordinary_burst','ordinary_temperature')]
    for candidate,reference in pairs:
        diff=np.log(np.clip(chosen[candidate],1e-10,1))-np.log(np.clip(chosen[reference],1e-10,1))
        key=candidate+'_vs_'+reference
        comparisons[key]={'tournament':ci(diff,d),'player_tournament':ci(diff,d,'player_tournament'),
                          'strata':{}}
        for label in sorted(set(labels)):
            mask=labels==label
            if mask.sum():comparisons[key]['strata'][label]=ci(diff[mask],d[mask])
    # Independent simulation replicate; no retuning after observing evaluation.
    convergence={}
    for cap in [None,CAP]:
        arm=arm_name(cap);rate=rates[arm]
        d2=season(2014,cap,True,seed=9029)
        p2=(1-rate)*d2.iid.to_numpy()+rate*d2.conditional_burst.to_numpy()
        p1=chosen[arm+'_burst']
        convergence[arm]={'seed1_log_loss':float(-np.log(p1).mean()),'seed2_log_loss':float(-np.log(p2).mean()),
                          'mean_absolute_probability_difference':float(np.abs(p1-p2).mean())}
        d[arm+'_burst_seed2']=p2
    for k,p in chosen.items():d[k]=p
    d['stratum']=labels
    d.to_csv(OUT/'comparison_2014_predictions.csv',index=False)
    result={'utc':utcnow(),'n':len(d),'tuning':curves,'rates':rates,'temperature':temperature,
            'metrics':metrics,'comparisons':comparisons,'simulation_check':convergence,
            'fit_diagnostics':fit_diagnostics(),
            'note':'Positive paired log-loss gain favors the candidate. Exploratory development comparison; no 2025 data.'}
    write_json(OUT/'comparison.json',result)
    report(result)
    return result


def report(r):
    lines=['# Independent full-model comparison','',f"Generated {r['utc']}; {r['n']} author-fixture matches in 2014. Tune on 2013.",'',
           'Same full Ingram ability model and priors; ordinary binomial objective versus Huber-deviance robust objective (cap 2.5).',
           'Historical deciding-set rules; posterior-averaged forecasts. Positive paired gain favors candidate.','',
           '| Method | Log loss (tournament 95% CI) | Accuracy | Brier | Cal. slope | ECE-10 |',
           '|---|---|---|---|---|---|']
    for name,m in r['metrics'].items():
        ll=m['log_loss']['tournament'];acc=m['accuracy']['tournament'];br=m['brier']['tournament'];cal=m['calibration']
        lines.append(f"| {name} | {ll['mean']:.5f} [{ll['lo']:.5f}, {ll['hi']:.5f}] | "
                     f"{acc['mean']:.3f} | {br['mean']:.4f} | {cal['slope']:.3f} | {cal['ece_10']:.4f} |")
    lines+=['','Paired comparisons:','']
    for name,c in r['comparisons'].items():
        t=c['tournament'];p=c['player_tournament']
        lines.append(f"- {name}: {t['mean']:+.5f}; tournament CI [{t['lo']:+.5f}, {t['hi']:+.5f}]; player-tournament CI [{p['lo']:+.5f}, {p['hi']:+.5f}].")
    lines+=['','Robust-estimation gain within the same forecast rule by appearance stratum:','',
            '| Stratum | Robust burst vs ordinary burst | Robust iid vs ordinary iid | n |',
            '|---|---:|---:|---:|']
    left=r['comparisons']['robust_2p5_burst_vs_ordinary_burst']['strata']
    right=r['comparisons']['robust_2p5_iid_vs_ordinary_iid']['strata']
    for label in ['both_first','mixed_first_later','both_later']:
        a=left[label];b=right[label]
        lines.append(f"| {label} | {a['mean']:+.5f} [{a['lo']:+.5f}, {a['hi']:+.5f}] | "
                     f"{b['mean']:+.5f} [{b['lo']:+.5f}, {b['hi']:+.5f}] | {a['n']} |")
    lines.append(f"\nUnknown appearance order ({left['unknown']['n']} round-robin rows) is omitted from stratum inference.")
    lines+=['',f"Frozen burst rates: {r['rates']}. Temperature control: {r['temperature']['chosen']}.",'',
            'Fit diagnostics: '+ '; '.join(
                f"{arm}: max R-hat {d['max_rhat']:.3f}, min bulk ESS {d['min_ess_bulk']:.0f}, "
                f"{d['divergences']} divergences, R-hat >1.01 in periods {d['periods_rhat_over_1p01']}"
                for arm,d in r['fit_diagnostics'].items())+'.','',
            'Player-tournament intervals use pigeonhole resampling of both player memberships; they are a sensitivity analysis.',
            'The author cohort excludes retirements. This tests a specified robust objective and burst forecast, not arbitrary contamination guarantees.',
            'Fits use bimonthly information cutoffs, so first/later-round differences do not measure within-tournament updating.',
            'See PROTOCOL.md, fit diagnostics, and comparison.json for the simulation replicate and full results.']
    (OUT/'comparison_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['baseline','compare','ordinary','robust'],default='baseline')
    args=ap.parse_args()
    sampler_progress()
    if args.stage=='baseline':baseline()
    elif args.stage=='compare':compare()
    elif args.stage=='ordinary':
        baseline()
        season(2013,None,True)
        season(2014,None,True)
    else:
        season(2013,CAP,True)
        season(2014,CAP,True)
