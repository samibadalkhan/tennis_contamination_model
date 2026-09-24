"""Post-hoc calibration control for the verified 2013 ordered-point pilot.

This does not alter the registered early-real comparison. It applies the same
temperature grid recorded in the independent full-model protocol, selects on
2012 match forecasts, and evaluates the frozen transform on saved 2013
ordinary forecasts. No 2025 file is read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src import trials
from src.experiments.early_real import (evaluate, learn, load_early, make_filter,
                                        surface_intercepts, clustered_interval)
from src.util import RESULTS, utcnow, write_json

TEMPERATURES = (.75, .9, 1., 1.1, 1.25, 1.5, 2.)
OUT = RESULTS / 'independent'


def loss(y, p):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    y=np.asarray(y,float)
    return -y*np.log(p)-(1-y)*np.log1p(-p)


def paired(frame, reference, candidate):
    d=frame[['tourney_id']].copy()
    d['gain']=np.asarray(reference)-np.asarray(candidate);d['n']=1
    return clustered_interval(d,'gain',['tourney_id'])


def run():
    matches,_=load_early()
    warm=[m for m in matches if m.year==2011]
    f=make_filter(surface_intercepts(warm),.03,'ordinary');cache={}
    for m in warm:learn(f,m,'ordinary',cache)
    _,validation=evaluate(f,[m for m in matches if m.year==2012],
                          'ordinary','online',set(f.sm),cache)
    curve=[]
    for temperature in TEMPERATURES:
        p=expit(logit(validation.p.clip(1e-9,1-1e-9))/temperature)
        value=float(loss(validation.y,p).mean())
        curve.append({'temperature':temperature,'match_logloss':value})
        trials.log(stage='independent-early-real-control',kind='temperature-2012',
                   params={'temperature':temperature,'sigma':.03},
                   metrics={'match_logloss':value},
                   note='Post-hoc control grid pre-existed in full-model protocol; no 2025 data.')
    chosen=min(curve,key=lambda x:x['match_logloss'])['temperature']
    saved=pd.read_csv(RESULTS/'early_real_2013'/'match_predictions.csv')
    modes={}
    for mode in ('frozen','online'):
        ordinary=saved[(saved.arm=='ordinary')&(saved['mode']==mode)].copy()
        huber=saved[(saved.arm=='huber_2.5')&(saved['mode']==mode)].copy()
        huber=huber.set_index('match_id').loc[ordinary.match_id].reset_index()
        p=expit(logit(ordinary.p.clip(1e-9,1-1e-9))/chosen)
        ordinary_loss=ordinary.loss_sum.to_numpy();huber_loss=huber.loss_sum.to_numpy()
        temperature_loss=loss(ordinary.y,p)
        modes[mode]={
            'ordinary_match_logloss':float(ordinary_loss.mean()),
            'temperature_match_logloss':float(temperature_loss.mean()),
            'huber_match_logloss':float(huber_loss.mean()),
            'temperature_vs_ordinary_gain':paired(ordinary,ordinary_loss,temperature_loss),
            'temperature_vs_huber_gain':paired(ordinary,huber_loss,temperature_loss),
            'matches':len(ordinary),
        }
    source=Path(__file__)
    result={'utc':utcnow(),'status':'post-hoc exploratory control',
            'validation_year':2012,'evaluation_year':2013,
            'temperature_grid':TEMPERATURES,'validation_curve':curve,
            'chosen_temperature':chosen,'modes':modes,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'note':'Grid was specified in the independent full-model protocol before this control. '
                   'It was not registered in the early-real protocol; interpret only as a calibration diagnostic. '
                   'The selected value is the maximum of the grid, so the optimum is not bracketed.'}
    write_json(OUT/'early_real_calibration_control.json',result)
    lines=['# Post-hoc calibration control for the early real-point result','',
           result['note'],'',
           f"The 2012 grid selected temperature **{chosen}**, the grid maximum, so the optimum is not bracketed. "
           'Accuracy and ranking are unchanged; only probability calibration changes.','',
           '| 2013 mode | Ordinary | Temperature | Huber 2.5 | Temperature gain vs ordinary [tournament CI] | Temperature gain vs Huber [tournament CI] |',
           '|---|---:|---:|---:|---:|---:|']
    for mode,d in modes.items():
        a=d['temperature_vs_ordinary_gain'];b=d['temperature_vs_huber_gain']
        lines.append(f"| {mode} | {d['ordinary_match_logloss']:.5f} | {d['temperature_match_logloss']:.5f} | "
                     f"{d['huber_match_logloss']:.5f} | {a['mean']:+.5f} [{a['lo']:+.5f}, {a['hi']:+.5f}] | "
                     f"{b['mean']:+.5f} [{b['lo']:+.5f}, {b['hi']:+.5f}] |")
    lines+=['',
            'Only four 2013 tournaments contribute to these intervals. The point estimate shows that ordinary-model '
            'overconfidence can more than explain the Huber match-log-loss advantage; the comparison with Huber remains '
            'statistically imprecise. This control does not change the registered point-log-loss result.']
    (OUT/'early_real_calibration_control.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    run()
