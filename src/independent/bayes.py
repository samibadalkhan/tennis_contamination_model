"""Full Ingram hierarchical model with NUTS; optional robust deviance objective.

The ordinary model follows the author's stan_model.stan, reimplemented in PyMC.
Robust fitting applies Huber loss to binomial deviance residuals, retaining the
same latent ability model and priors. This is a generalized posterior, not a
claim to reproduce MMW or a normalized non-iid observation likelihood.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('PYTENSOR_FLAGS', 'base_compiledir=/private/tmp/tennis_pytensor')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tennis_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/tennis_cache')

import numpy as np
import pandas as pd
from scipy.special import expit, xlogy

from src import trials
from src.util import DATA, RESULTS, write_json, utcnow

OUT = RESULTS / 'independent'
HYPERS = ['intercept','sigma_s0','sigma_r0','sigma_s','sigma_r','sigma_surf','sigma_t']


def reference_data():
    root = DATA / 'independent_ingram'
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    f = next(f for f in manifest['files'] if f['name'] == 'dataset.csv')
    assert hashlib.sha256((root / f['name']).read_bytes()).hexdigest() == f['sha256']
    raw = pd.read_csv(root / 'dataset.csv')
    d = pd.DataFrame({
        'p1': raw.winner, 'p2': raw.loser,
        'date': pd.to_datetime(raw.start_date), 'year': raw.year,
        'surface': raw.surface, 'tournament': raw.tournament_name,
        'round': raw['round'], 'score': raw.score,
        'n1': raw.loser_return_points_total,
        'k1': raw.loser_return_points_total - raw.loser_return_points_won,
        'n2': raw.winner_return_points_total,
        'k2': raw.winner_return_points_total - raw.winner_return_points_won,
    })
    assert d.year.max() == 2014, 'Author fixture changed; review before running'
    d['period'] = ((d.date.dt.year - 2011) * 12 + d.date.dt.month - 1) // 2
    d['match_id'] = [f'author-{i}' for i in d.index]
    d['tourney_id'] = d.year.astype(str) + '-' + d.tournament
    d['best_of'] = np.where(d.tournament.isin(['Australian Open','FO - RG','Wimbledon','US Open']),5,3)
    for side in [1,2]:
        assert (d[f'n{side}'] > 0).all()
        assert ((d[f'k{side}'] >= 0) & (d[f'k{side}'] <= d[f'n{side}'])).all()
    # The author fixture is already the completed-match cohort used in the paper.
    assert len(d[d.year == 2014]) == 2208
    return d


def build_model(train, cap=None):
    import pymc as pm
    import pytensor.tensor as pt
    players = sorted(set(train.p1) | set(train.p2))
    surfaces = sorted(train.surface.unique())
    tournaments = sorted(train.tournament.unique())
    enc = {'players': players, 'surfaces': surfaces, 'tournaments': tournaments}
    pmap = {p:i for i,p in enumerate(players)}
    smap = {s:i for i,s in enumerate(surfaces)}
    tmap = {t:i for i,t in enumerate(tournaments)}
    a = train.p1.map(pmap).to_numpy(); b = train.p2.map(pmap).to_numpy()
    srv = np.r_[a,b]; ret = np.r_[b,a]
    surface = np.tile(train.surface.map(smap).to_numpy(),2)
    tournament = np.tile(train.tournament.map(tmap).to_numpy(),2)
    period = np.tile(train.period.to_numpy(),2)
    k = np.r_[train.k1,train.k2].astype('int64')
    n = np.r_[train.n1,train.n2].astype('int64')
    P = len(players); T = int(train.period.max()) + 1
    # Integrate out unobserved random-walk states exactly. This preserves the
    # author's prior while avoiding thousands of uninformed NUTS dimensions.
    state_keys=sorted(set(zip(np.r_[a,b],period)))
    state_map={key:i for i,key in enumerate(state_keys)}
    first=[];gap=[];group_start=[];last=[];last_period=[]
    for p in range(P):
        indices=[i for i,(q,t) in enumerate(state_keys) if q==p]
        previous=0
        for j,i in enumerate(indices):
            t=state_keys[i][1]
            first.append(j==0);gap.append(t if j==0 else t-previous)
            group_start.append(indices[0]);previous=t
        last.append(indices[-1]);last_period.append(previous)
    enc['last_period']=last_period
    first=np.array(first);gap=np.array(gap)
    group_start=np.array(group_start);last=np.array(last)
    s_index=np.array([state_map[p,t] for p,t in zip(srv,period)])
    r_index=np.array([state_map[p,t] for p,t in zip(ret,period)])
    with pm.Model() as model:
        intercept = pm.Normal('intercept',0,1)
        sigmas = {name:pm.HalfNormal(name,1) for name in HYPERS[1:]}
        zs = pm.Normal('eta_s',0,1,shape=len(state_keys))
        zr = pm.Normal('eta_r',0,1,shape=len(state_keys))
        ss0=pt.cumsum(zs*pt.sqrt(first*sigmas['sigma_s0']**2+gap*sigmas['sigma_s']**2))
        rr0=pt.cumsum(zr*pt.sqrt(first*sigmas['sigma_r0']**2+gap*sigmas['sigma_r']**2))
        ss=ss0-pt.concatenate([pt.zeros(1),ss0])[group_start]
        rr=rr0-pt.concatenate([pt.zeros(1),rr0])[group_start]
        surf = pm.Deterministic('surf',pm.Normal('eta_surf',0,1,shape=(len(surfaces),P))*sigmas['sigma_surf'])
        tour = pm.Deterministic('tour',pm.Normal('eta_tour',0,1,shape=len(tournaments))*sigmas['sigma_t'])
        pm.Deterministic('last_s',ss[last]); pm.Deterministic('last_r',rr[last])
        logits = intercept + ss[s_index] - rr[r_index] + surf[surface,srv] - surf[surface,ret] + tour[tournament]
        if cap is None:
            pm.Binomial('observed',n=n,logit_p=logits,observed=k)
        else:
            saturated = xlogy(k,k/n) + xlogy(n-k,1-k/n)
            half_dev = pt.maximum(saturated - (k*logits - n*pt.softplus(logits)),1e-12)
            rho = pt.where(half_dev <= cap**2/2,half_dev,cap*pt.sqrt(2*half_dev)-cap**2/2)
            pm.Potential('robust_binomial_deviance',-pt.sum(rho))
    return model, enc


def fit(period, cap=None, draws=1000, tune=1000, chains=4, seed=521, smoke=False):
    import pymc as pm
    import arviz as az
    data = reference_data()
    assert 1 <= period <= 23
    train = data[data.period < period]
    arm = 'ordinary' if cap is None else f'robust_{cap}'.replace('.', 'p')
    folder = OUT / ('smoke' if smoke else 'fits')
    folder.mkdir(parents=True,exist_ok=True)
    stem = folder / f'{arm}_period_{period}'
    config = dict(period=period,cap=cap,draws=draws,tune=tune,chains=chains,seed=seed,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if stem.with_suffix('.json').exists():
        old = json.loads(stem.with_suffix('.json').read_text())
        if old['config'] == config and stem.with_suffix('.npz').exists():
            print('CACHED',stem,flush=True)
            return old
        raise ValueError(f'Existing different fit at {stem}; use a new output path instead of overwriting')
    print('FIT',arm,period,'matches',len(train),'draws',draws,'chains',chains,flush=True)
    started = time.monotonic()
    model, enc = build_model(train,cap)
    with model:
        trace = pm.sample(draws=draws,tune=tune,chains=chains,cores=min(chains,4),
                          random_seed=seed+period,target_accept=0.9,nuts_sampler='nutpie',
                          progressbar=False,blas_cores=1)
    posterior = trace['posterior']
    arrays = {}
    for name in HYPERS + ['last_s','last_r','surf','tour']:
        ar = posterior[name].values
        arrays[name] = ar.reshape((-1,)+ar.shape[2:])
    np.savez_compressed(stem.with_suffix('.npz'),**arrays)
    summary = az.summary(trace,var_names=HYPERS+['last_s','last_r','surf','tour'])
    summary.to_csv(str(stem)+'_diagnostics.csv')
    diag = {'max_rhat': float(summary.r_hat.max()), 'min_ess_bulk':float(summary.ess_bulk.min()),
            'divergences':int(trace['sample_stats']['diverging'].values.sum()),
            'hyperparameters':summary.loc[HYPERS].reset_index().to_dict('records')}
    out = {'utc':utcnow(),'config':config,'encoding':enc,'n_train':len(train),
           'diagnostics':diag,'seconds':time.monotonic()-started,
           'period_start':str(pd.Timestamp(2011+period//6,1+(period%6)*2,1).date())}
    write_json(stem.with_suffix('.json'),out)
    trials.log(stage='independent-full-bayes',kind='smoke' if smoke else 'fit',params=config,
               metrics={'max_rhat':diag['max_rhat'],'min_ess_bulk':diag['min_ess_bulk'],
                        'divergences':diag['divergences'],'seconds':out['seconds']})
    print('DONE',arm,period,diag['max_rhat'],diag['min_ess_bulk'],'seconds',round(out['seconds']),flush=True)
    return out


def serve_draws(rows, arrays, enc, seed=99):
    """Joint posterior-predictive serve probabilities, including next-period drift."""
    rng = np.random.default_rng(seed)
    N = len(arrays['intercept'])
    ps = list(enc['players']); ts = list(enc['tournaments']); fs = list(enc['surfaces'])
    allp = sorted(set(rows.p1)|set(rows.p2))
    skills = {}
    for p in allp:
        if p in ps:
            i=ps.index(p)
            gap=float(rows.period.iloc[0]-enc['last_period'][i])
            s=arrays['last_s'][:,i]+rng.normal(size=N)*arrays['sigma_s']*np.sqrt(gap)
            r=arrays['last_r'][:,i]+rng.normal(size=N)*arrays['sigma_r']*np.sqrt(gap)
        else:
            s=rng.normal(size=N)*arrays['sigma_s0']
            r=rng.normal(size=N)*arrays['sigma_r0']
        skills[p]=(s,r)
    sf={}
    for p in allp:
        for f in rows.surface.unique():
            sf[p,f]=(arrays['surf'][:,fs.index(f),ps.index(p)] if p in ps and f in fs
                     else rng.normal(size=N)*arrays['sigma_surf'])
    tf={t:(arrays['tour'][:,ts.index(t)] if t in ts else rng.normal(size=N)*arrays['sigma_t'])
        for t in rows.tournament.unique()}
    pa=[];pb=[]
    for row in rows.itertuples():
        shared=arrays['intercept']+tf[row.tournament]
        surface=sf[row.p1,row.surface]-sf[row.p2,row.surface]
        pa.append(expit(shared+skills[row.p1][0]-skills[row.p2][1]+surface))
        pb.append(expit(shared+skills[row.p2][0]-skills[row.p1][1]-surface))
    return np.array(pa),np.array(pb)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--period',type=int,default=18)
    ap.add_argument('--cap',type=float)
    ap.add_argument('--draws',type=int,default=1000)
    ap.add_argument('--tune',type=int,default=1000)
    ap.add_argument('--chains',type=int,default=4)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    fit(a.period,a.cap,a.draws,a.tune,a.chains,smoke=a.smoke)
