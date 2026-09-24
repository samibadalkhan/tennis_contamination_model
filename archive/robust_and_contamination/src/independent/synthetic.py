"""Fresh point-generated brackets to check recovery with the full ability model.

Exploratory finite-world checks, not a replacement verdict for the old gate.
Permanent-jump final error is reported; no claim of bounded tracking lag.
"""
import os
os.environ.setdefault('PYTENSOR_FLAGS','base_compiledir=/private/tmp/tennis_pytensor')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tennis_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/tennis_cache')
import json
import numpy as np
import pandas as pd
from numba import njit
from scipy.special import expit,logit

from src import trials
from src.util import write_json,utcnow
from src.independent.bayes import OUT,build_model
from src.independent.scoring import play


@njit(cache=True)
def seeded_match(pa,pb,seed,start_a,end_a,sa,start_b,end_b,sb):
    np.random.seed(seed)
    return play(pa,pb,3,6,7,start_a,end_a,sa,start_b,end_b,sb)


def world(generator,condition,seed):
    rng=np.random.default_rng(7000+seed)
    P,T=16,4
    s0=rng.normal(0,.33,P);r0=rng.normal(0,.33,P)
    if generator=='rw':
        s=s0[:,None]+np.c_[np.zeros(P),np.cumsum(rng.normal(0,.05,(P,T-1)),axis=1)]
        r=r0[:,None]+np.c_[np.zeros(P),np.cumsum(rng.normal(0,.05,(P,T-1)),axis=1)]
    else:
        phase=rng.uniform(0,2*np.pi,(P,1));t=np.arange(T)[None,:]
        s=s0[:,None]+.2*np.sin(phase+t*.6)
        r=r0[:,None]+.2*np.cos(phase+t*.6)
    if condition=='permanent_jump':
        ids=rng.choice(P,4,replace=False);s[ids,2:]-=.5;r[ids,2:]-=.5
    rows=[]
    for period in range(T):
        for event in range(3):
            impaired=(rng.random(P)<.15) if condition=='tournament' else np.zeros(P,bool)
            alive=rng.permutation(P).tolist();rnd=0
            while len(alive)>1:
                advance=[]
                for i in range(0,len(alive),2):
                    a,b=alive[i:i+2]
                    sev_a=1.5*impaired[a];sev_b=1.5*impaired[b]
                    start_a=start_b=0;end_a=end_b=100000
                    if condition=='point_burst' and rng.random()<.2:
                        start=int(rng.integers(81));end=start+40
                        if rng.random()<.5:start_a,end_a,sev_a=start,end,1.5
                        else:start_b,end_b,sev_b=start,end,1.5
                    pa=expit(logit(.64)+s[a,period]-r[b,period])
                    pb=expit(logit(.64)+s[b,period]-r[a,period])
                    aw,ka,na,kb,nb,npts=seeded_match(pa,pb,int(rng.integers(2**30)),
                                                  start_a,end_a,sev_a,start_b,end_b,sev_b)
                    win,lose=(a,b) if aw else (b,a)
                    k1,n1,k2,n2=(ka,na,kb,nb) if aw else (kb,nb,ka,na)
                    rows.append(dict(p1=f'p{win:02}',p2=f'p{lose:02}',period=period,
                                     surface='hard',tournament='synthetic',k1=k1,n1=n1,k2=k2,n2=n2,
                                     round=rnd,point_count=npts))
                    advance.append(win)
                alive=advance;rnd+=1
    return pd.DataFrame(rows),s[:,-1],r[:,-1]


def run():
    import pymc as pm
    import arviz as az
    cells=[];folder=OUT/'synthetic';folder.mkdir(parents=True,exist_ok=True)
    for generator in ['rw','arc']:
        for condition in ['none','point_burst','tournament','permanent_jump']:
            for seed in range(2):
                path=folder/f'{generator}_{condition}_{seed}.json'
                if path.exists():cells.append(json.loads(path.read_text()));continue
                frame,true_s,true_r=world(generator,condition,seed)
                result={'generator':generator,'condition':condition,'seed':seed,'n':len(frame),'arms':{}}
                for cap in [None,2.5]:
                    arm='ordinary' if cap is None else 'robust_2p5'
                    print('SYNTHETIC',generator,condition,seed,arm,flush=True)
                    model,enc=build_model(frame,cap)
                    with model:
                        trace=pm.sample(draws=300,tune=500,chains=2,cores=1,random_seed=8000+seed,
                                        target_accept=.9,nuts_sampler='nutpie',progressbar=False,blas_cores=1)
                    post=trace['posterior']
                    es=post['last_s'].values.mean(axis=(0,1));er=post['last_r'].values.mean(axis=(0,1))
                    es-=es.mean();er-=er.mean()
                    ts=true_s-true_s.mean();tr=true_r-true_r.mean()
                    errors=np.sqrt((es-ts)**2+(er-tr)**2)
                    diag=az.summary(trace,var_names=['last_s','last_r'])
                    result['arms'][arm]={'rmse':float(np.sqrt(np.mean(errors**2))),
                                         'per_player_error':dict(zip(enc['players'],errors.tolist())),
                                         'max_rhat':float(diag.r_hat.max()),
                                         'min_ess_bulk':float(diag.ess_bulk.min()),
                                         'divergences':int(trace['sample_stats']['diverging'].values.sum())}
                    trials.log(stage='independent-full-bayes',kind='synthetic-recovery',
                               params={'generator':generator,'condition':condition,'seed':seed,'cap':cap},
                               metrics=result['arms'][arm])
                result['gain']=result['arms']['ordinary']['rmse']-result['arms']['robust_2p5']['rmse']
                write_json(path,result);cells.append(result)
    diagnostics={}
    for arm in ['ordinary','robust_2p5']:
        values=[c['arms'][arm] for c in cells]
        diagnostics[arm]={
            'max_rhat':max(v['max_rhat'] for v in values),
            'min_ess_bulk':min(v['min_ess_bulk'] for v in values),
            'total_divergences':sum(v['divergences'] for v in values),
            'failed_cells':sum(bool(v['max_rhat']>1.05 or v['divergences']>0) for v in values),
            'total_cells':len(values),
        }
    valid=all(v['failed_cells']==0 for v in diagnostics.values())
    summary={'utc':utcnow(),'cells':cells,'diagnostics':diagnostics,'valid':valid,
             'note':'Two seeds per generator: exploratory recovery checks, not a precise power assessment or bounded-lag gate. '
                    'The aggregate RMSE table is invalid for inference when any fit diagnostic fails.'}
    write_json(OUT/'synthetic_recovery.json',summary)
    lines=['# Independent coherent synthetic recovery','',summary['note'],'',
           f"**Diagnostic verdict: {'PASS' if valid else 'FAIL — do not interpret the RMSE differences as evidence.'}**",'',
           f"Ordinary: {diagnostics['ordinary']['failed_cells']}/{diagnostics['ordinary']['total_cells']} failed cells, "
           f"max R-hat {diagnostics['ordinary']['max_rhat']:.3f}, {diagnostics['ordinary']['total_divergences']} divergences.  ",
           f"Robust: {diagnostics['robust_2p5']['failed_cells']}/{diagnostics['robust_2p5']['total_cells']} failed cells, "
           f"max R-hat {diagnostics['robust_2p5']['max_rhat']:.3f}, {diagnostics['robust_2p5']['total_divergences']} divergences.",'',
           '| Generator | Condition | Ordinary RMSE | Robust RMSE | Paired gain |',
           '|---|---|---|---|---|']
    for generator in ['rw','arc']:
        for condition in ['none','point_burst','tournament','permanent_jump']:
            selected=[c for c in cells if c['generator']==generator and c['condition']==condition]
            a=np.mean([c['arms']['ordinary']['rmse'] for c in selected])
            b=np.mean([c['arms']['robust_2p5']['rmse'] for c in selected])
            lines.append(f'| {generator} | {condition} | {a:.4f} | {b:.4f} | {a-b:+.4f} |')
    (OUT/'synthetic_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
