"""Generate the consolidated figure set for every completed result family.

Reads saved result artifacts only. It does not refit models or read 2025 data.

    MPLCONFIGDIR=/private/tmp/tennis_plots python -m results.plots.make_all_results_plots
"""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tennis_all_results_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/tennis_cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results'/'plots'/'all_results'
OUT.mkdir(parents=True,exist_ok=True)

C={'ordinary':'#3567A8','robust':'#D97732','burst':'#2A9D8F','temperature':'#7B5AA6',
   'block':'#008C95','oracle':'#6F7782','negative':'#C44E52','ink':'#242424',
   'muted':'#777777','grid':'#E4E7EB','light':'#F3F5F7'}

plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'#C7CCD1',
 'axes.grid':True,'grid.color':C['grid'],'grid.linewidth':.7,'axes.spines.top':False,
 'axes.spines.right':False,'font.size':10,'axes.titleweight':'bold','figure.dpi':120,
 'savefig.dpi':180})


def load(path):
    with open(ROOT/path) as f:return json.load(f)


def save(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def errorbarh(ax,y,d,color,label=None,marker='o'):
    ax.errorbar(d['mean'],y,xerr=[[d['mean']-d['lo']],[d['hi']-d['mean']]],
                color=color,fmt=marker,capsize=4,lw=1.8,ms=6,label=label,zorder=3)


def executive_summary():
    r=load('results/independent/comparison.json')
    e=load('results/early_real_2013/results.json')
    names=['Ordinary iid','Robust iid','Ordinary + burst','Robust + burst','Ordinary + temp.']
    keys=['ordinary_iid','robust_2p5_iid','ordinary_burst','robust_2p5_burst','ordinary_temperature']
    colors=[C['ordinary'],C['robust'],C['burst'],C['robust'],C['temperature']]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    vals=[r['metrics'][k]['log_loss']['tournament']['mean'] for k in keys]
    axes[0].barh(np.arange(5),vals,color=colors,height=.64)
    axes[0].set_yticks(np.arange(5),names);axes[0].invert_yaxis();axes[0].set_xlim(.575,.598)
    axes[0].axvline(.592,color=C['ink'],ls=':',lw=1.2,label='Ingram 0.592')
    for y,v in enumerate(vals):axes[0].text(v+.00025,y,f'{v:.5f}',va='center',fontsize=9)
    axes[0].set_xlabel('2014 log loss (lower is better)');axes[0].set_title('Full Bayesian model')
    axes[0].legend(frameon=False,loc='lower right',fontsize=8.5)

    pairs=[('Robust iid','robust_2p5_iid_vs_ordinary_iid'),
           ('Robust + burst','robust_2p5_burst_vs_ordinary_burst'),
           ('Ordinary + burst','ordinary_burst_vs_ordinary_iid')]
    for y,(label,key) in enumerate(pairs):
        errorbarh(axes[1],y,r['comparisons'][key]['tournament'],
                  C['burst'] if 'Ordinary' in label else C['robust'])
    axes[1].axvline(0,color=C['ink'],lw=1,ls='--')
    axes[1].set_yticks(range(3),[p[0] for p in pairs]);axes[1].invert_yaxis()
    axes[1].set_xlabel('paired log-loss gain');axes[1].set_title('What improves over ordinary iid?')
    axes[1].text(.02,.03,'Positive = better',transform=axes[1].transAxes,color=C['muted'],fontsize=8.5)

    rows=[]
    for mode in ('frozen','online'):
        for arm,label,color in [('huber_2.5','Huber',C['robust']),('block_16','Block-16',C['block'])]:
            rows.append((f'{label}\n{mode}',e['summary'][mode][arm]['point_gain'],color))
    for y,(label,d,color) in enumerate(rows):errorbarh(axes[2],y,d,color)
    axes[2].axvline(0,color=C['ink'],lw=1,ls='--');axes[2].set_yticks(range(4),[x[0] for x in rows])
    axes[2].invert_yaxis();axes[2].set_xlabel('paired point-log-loss gain')
    axes[2].set_title('2013 ordered real points')
    fig.suptitle('Robust tennis modeling: results at a glance',fontsize=15,y=1.04)
    fig.text(.5,-.02,'Full model: robust estimation ties ordinary; calibration helps. Real ordered points: Huber is tiny-positive, block filtering is tiny-negative.',
             ha='center',fontsize=10,color=C['ink'])
    fig.tight_layout()
    save(fig,'01_executive_summary.png')


def full_scores_calibration():
    r=load('results/independent/comparison.json')
    pred=pd.read_csv(ROOT/'results/independent/comparison_2014_predictions.csv')
    order=[('ordinary_iid','Ordinary iid',C['ordinary']),('robust_2p5_iid','Robust iid',C['robust']),
           ('ordinary_burst','Ordinary + burst',C['burst']),('robust_2p5_burst','Robust + burst','#B85C25'),
           ('ordinary_temperature','Ordinary + temperature',C['temperature'])]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for y,(key,label,color) in enumerate(order):
        d=r['metrics'][key]['log_loss']['tournament'];errorbarh(axes[0],y,d,color)
        axes[0].text(d['mean']+.0006,y,f"{d['mean']:.5f}",va='center',fontsize=8.5)
    axes[0].axvline(.592,color=C['ink'],ls=':',lw=1.2,label='Published 0.592')
    axes[0].set_yticks(range(len(order)),[x[1] for x in order]);axes[0].invert_yaxis()
    axes[0].set_xlabel('2014 log loss with tournament-bootstrap 95% CI')
    axes[0].set_title('Forecast score');axes[0].legend(frameon=False,fontsize=8.5)
    bins=np.linspace(0,1,11)
    axes[1].plot([0,1],[0,1],color=C['ink'],ls='--',lw=1,label='Perfect calibration')
    for key,label,color in [order[0],order[2],order[4],order[3]]:
        p=pred[key].to_numpy();q=np.r_[p,1-p];y=np.r_[np.ones(len(p)),np.zeros(len(p))]
        ids=np.minimum(np.digitize(q,bins)-1,9);xs=[];ys=[]
        for b in range(10):
            m=ids==b
            if m.sum()>=20:xs.append(q[m].mean());ys.append(y[m].mean())
        axes[1].plot(xs,ys,marker='o',ms=4,lw=1.6,color=color,label=label)
    axes[1].set_xlim(.18,.82);axes[1].set_ylim(.18,.82);axes[1].set_aspect('equal',adjustable='box')
    axes[1].set_xlabel('mean forecast probability');axes[1].set_ylabel('observed frequency')
    axes[1].set_title('Reliability (winner-first rows mirrored)');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Full-model scoring and calibration · 2,208 matches in 2014',fontsize=14,y=1.02)
    fig.tight_layout();save(fig,'02_full_model_scores_and_calibration.png')


def full_paired_gains():
    r=load('results/independent/comparison.json')
    items=[('Robust + burst vs ordinary iid','robust_2p5_burst_vs_ordinary_iid'),
           ('Ordinary + burst vs ordinary iid','ordinary_burst_vs_ordinary_iid'),
           ('Robust iid vs ordinary iid','robust_2p5_iid_vs_ordinary_iid'),
           ('Robust + burst vs ordinary + burst','robust_2p5_burst_vs_ordinary_burst'),
           ('Ordinary + burst vs temperature','ordinary_burst_vs_ordinary_temperature')]
    fig,ax=plt.subplots(figsize=(9.5,5.2));ys=np.arange(len(items))
    for y,(label,key) in zip(ys,items):
        d=r['comparisons'][key]
        errorbarh(ax,y-.11,d['tournament'],C['ordinary'],label='Tournament bootstrap' if y==0 else None)
        errorbarh(ax,y+.11,d['player_tournament'],C['robust'],label='Player–tournament sensitivity' if y==0 else None,marker='s')
    ax.axvline(0,color=C['ink'],ls='--',lw=1);ax.set_yticks(ys,[x[0] for x in items]);ax.invert_yaxis()
    ax.set_xlabel('paired log-loss gain (positive favors first method)')
    ax.set_title('Full Bayesian comparison: paired uncertainty')
    ax.legend(frameon=False,loc='lower right');fig.tight_layout();save(fig,'03_full_model_paired_gains.png')


def tuning_and_simulation():
    r=load('results/independent/comparison.json')
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for arm,label,color in [('ordinary','Ordinary',C['ordinary']),('robust_2p5','Robust 2.5',C['robust'])]:
        x=[z['rate'] for z in r['tuning'][arm]];y=[z['log_loss'] for z in r['tuning'][arm]]
        axes[0].plot(x,y,marker='o',color=color,lw=2,label=label)
        chosen=r['rates'][arm];z=next(v['log_loss'] for v in r['tuning'][arm] if v['rate']==chosen)
        axes[0].scatter([chosen],[z],s=110,facecolors='none',edgecolors=color,lw=2)
    axes[0].set_xlabel('potential-burst rate');axes[0].set_ylabel('2013 validation log loss')
    axes[0].set_title('Burst-rate tuning');axes[0].legend(frameon=False)
    curve={float(k):v for k,v in r['temperature']['curve'].items()};x=sorted(curve);y=[curve[z] for z in x]
    axes[1].plot(x,y,marker='o',color=C['temperature'],lw=2)
    t=r['temperature']['chosen'];axes[1].scatter([t],[curve[t]],s=110,facecolors='none',edgecolors=C['temperature'],lw=2)
    axes[1].set_xlabel('temperature');axes[1].set_ylabel('2013 validation log loss');axes[1].set_title('Calibration tuning')
    arms=['ordinary','robust_2p5'];x=np.arange(2);width=.34
    a=[r['simulation_check'][k]['seed1_log_loss'] for k in arms]
    b=[r['simulation_check'][k]['seed2_log_loss'] for k in arms]
    axes[2].bar(x-width/2,a,width,color=C['ordinary'],label='Seed 4001')
    axes[2].bar(x+width/2,b,width,color=C['burst'],label='Seed 9029')
    axes[2].set_xticks(x,['Ordinary','Robust 2.5']);axes[2].set_ylim(.5795,.5815)
    axes[2].set_ylabel('2014 burst-forecast log loss');axes[2].set_title('Monte Carlo replicate')
    axes[2].legend(frameon=False,fontsize=8.5)
    fig.suptitle('Frozen tuning and numerical sensitivity',fontsize=14,y=1.02)
    fig.tight_layout();save(fig,'04_tuning_and_simulation.png')


def diagnostics():
    r=load('results/independent/comparison.json')['fit_diagnostics']
    fig,axes=plt.subplots(2,1,figsize=(10,6.8),sharex=True)
    for arm,label,color in [('ordinary','Ordinary',C['ordinary']),('robust_2p5','Robust 2.5',C['robust'])]:
        p=r[arm]['periods'];x=[v['period'] for v in p]
        axes[0].plot(x,[v['max_rhat'] for v in p],marker='o',color=color,lw=1.8,label=label)
        axes[1].plot(x,[v['min_ess_bulk'] for v in p],marker='o',color=color,lw=1.8,label=label)
    axes[0].axhline(1.01,color=C['negative'],ls='--',lw=1,label='Strict 1.01 guide')
    axes[0].axhline(1.05,color=C['muted'],ls=':',lw=1,label='1.05')
    axes[0].set_ylabel('maximum R-hat');axes[0].set_title('Worst convergence statistic per rolling fit')
    axes[0].legend(frameon=False,ncol=4,fontsize=8.5)
    axes[1].axhline(400,color=C['muted'],ls='--',lw=1,label='ESS 400 guide')
    axes[1].set_ylabel('minimum bulk ESS');axes[1].set_xlabel('two-month period (12–17 = 2013; 18–23 = 2014)')
    axes[1].set_title('Lowest effective sample size per rolling fit');axes[1].legend(frameon=False)
    fig.suptitle('Full-model NUTS diagnostics · zero divergences in all 24 fits',fontsize=14,y=1.01)
    fig.tight_layout();save(fig,'05_sampling_diagnostics.png')


def early_real():
    e=load('results/early_real_2013/results.json');c=load('results/independent/early_real_calibration_control.json')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    rows=[]
    for mode in ('frozen','online'):
        for arm,label,color in [('huber_2.5','Huber',C['robust']),('block_16','Block-16',C['block'])]:
            rows.append((f'{label} · {mode}',e['summary'][mode][arm]['point_gain'],color))
    for y,(label,d,color) in enumerate(rows):errorbarh(axes[0],y,d,color)
    axes[0].axvline(0,color=C['ink'],ls='--',lw=1);axes[0].set_yticks(range(4),[r[0] for r in rows]);axes[0].invert_yaxis()
    axes[0].set_xlabel('point-log-loss gain');axes[0].set_title('Primary point forecast')
    modes=['frozen','online'];x=np.arange(2);width=.23
    for j,(arm,label,color) in enumerate([('ordinary','Ordinary',C['ordinary']),('huber_2.5','Huber',C['robust']),('block_16','Block-16',C['block'])]):
        axes[1].bar(x+(j-1)*width,[e['summary'][m][arm]['match_logloss']['mean'] for m in modes],width,color=color,label=label)
    axes[1].set_xticks(x,['Frozen','Online']);axes[1].set_ylabel('2013 match log loss');axes[1].set_title('Registered iid match forecast')
    axes[1].legend(frameon=False,fontsize=8)
    for j,(label,key,color) in enumerate([('Ordinary','ordinary_match_logloss',C['ordinary']),('Huber','huber_match_logloss',C['robust']),('Temperature','temperature_match_logloss',C['temperature'])]):
        axes[2].bar(x+(j-1)*width,[c['modes'][m][key] for m in modes],width,color=color,label=label)
    axes[2].set_xticks(x,['Frozen','Online']);axes[2].set_ylabel('2013 match log loss');axes[2].set_title('Post-hoc calibration control')
    axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Ordered real points · train 2011–2012, evaluate 2013',fontsize=14,y=1.02)
    fig.text(.5,-.025,'Point intervals use 361 server–event clusters; match results have only four tournament clusters.',ha='center',fontsize=9,color=C['muted'])
    fig.tight_layout();save(fig,'06_early_real_point_and_match.png')


def pilot_recovery():
    r=load('results/point_robust_pilot/results.json')['summary']
    generators=['rw','smooth'];conditions=['clean','burst','permanent_drop','permanent_rise']
    arms=[('huber_2.5','Huber',C['robust']),('block_8','Block-8','#65B8BE'),('block_16','Block-16',C['block']),
          ('block_32','Block-32','#005F68'),('oracle','Oracle',C['oracle'])]
    fig,axes=plt.subplots(2,2,figsize=(13,8.2),sharex=True)
    for ax,condition in zip(axes.flat,conditions):
        yt=[];pos=[];y=0
        for gen in generators:
            for arm,label,color in arms:
                d=next(v['focal_gain'] for v in r if v['generator']==gen and v['condition']==condition and v['arm']==arm)
                errorbarh(ax,y,d,color);yt.append(f'{gen.upper()} · {label}');pos.append(y);y+=1
            y+=.55
        ax.axvline(0,color=C['ink'],ls='--',lw=1);ax.set_yticks(pos,yt,fontsize=8);ax.invert_yaxis()
        ax.set_title(condition.replace('_',' ').title());ax.set_xlabel('affected-player RMSE reduction vs ordinary')
    fig.suptitle('Ordered-point synthetic pilot · 20 paired worlds per cell',fontsize=14,y=1.01)
    fig.text(.5,-.01,'Positive = better latent-skill recovery. The oracle is a known-mask reference.',ha='center',fontsize=9,color=C['muted'])
    fig.tight_layout();save(fig,'07_point_pilot_recovery.png')


def pilot_filtering():
    r=load('results/point_robust_pilot/results.json')['summary']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4));width=.22;x=np.arange(3)
    for j,(gen,label,color) in enumerate([('rw','Random walk',C['ordinary']),('smooth','Smooth',C['robust'])]):
        clean=[];bad=[]
        for size in (8,16,32):
            clean.append(next(v['removed_clean_fraction']['mean'] for v in r if v['generator']==gen and v['condition']=='burst' and v['arm']==f'block_{size}'))
            bad.append(next(v['removed_corrupted_fraction']['mean'] for v in r if v['generator']==gen and v['condition']=='burst' and v['arm']==f'block_{size}'))
        axes[0].bar(x+(j-.5)*width,clean,width,color=color,label=label)
        axes[1].bar(x+(j-.5)*width,bad,width,color=color,label=label)
    for ax in axes:ax.set_xticks(x,['Block 8','Block 16','Block 32']);ax.legend(frameon=False,fontsize=8.5)
    axes[0].set_ylabel('fraction removed');axes[0].set_title('Clean points removed in burst worlds')
    axes[1].set_ylabel('fraction removed');axes[1].set_title('Disrupted points removed')
    fig.suptitle('What the synthetic block filter actually removes',fontsize=14,y=1.02)
    fig.tight_layout();save(fig,'08_point_pilot_filtering.png')


def legacy_pipeline():
    robust=load('results/robust_check/compare_2014.json');sim=load('results/sim_forecast/sim_forecast_2014.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    names=['Ordinary iid','Huber 1.345 iid','Huber 2.5 iid','Ordinary burst','Huber 1.345 burst','Huber 2.5 burst']
    vals=[robust['arms']['ordinary']['log_loss'],robust['arms']['robust_1.345']['log_loss'],robust['arms']['robust_2.5']['log_loss'],
          sim['arms']['ordinary']['log_loss'],sim['arms']['robust_1.345']['log_loss'],sim['arms']['robust_2.5']['log_loss']]
    colors=[C['ordinary'],C['negative'],C['robust'],C['burst'],C['negative'],C['robust']]
    axes[0].barh(range(6),vals,color=colors);axes[0].set_yticks(range(6),names);axes[0].invert_yaxis();axes[0].set_xlim(.59,.625)
    for y,v in enumerate(vals):axes[0].text(v+.0005,y,f'{v:.4f}',va='center',fontsize=8.5)
    axes[0].set_xlabel('2014 log loss');axes[0].set_title('Simplified online-filter scores')
    items=[]
    for key,label in [('robust_1.345','Huber 1.345 iid'),('robust_2.5','Huber 2.5 iid')]:items.append((label,robust['comparisons'][key]['paired_ll_gain']))
    for key,label in [('robust_1.345','Huber 1.345 burst'),('robust_2.5','Huber 2.5 burst')]:items.append((label,sim['arms'][key]['paired_gain_vs_ordinary_sim']))
    for y,(label,d) in enumerate(items):errorbarh(axes[1],y,d,C['negative'] if '1.345' in label else C['robust'])
    axes[1].axvline(0,color=C['ink'],ls='--',lw=1);axes[1].set_yticks(range(4),[v[0] for v in items]);axes[1].invert_yaxis()
    axes[1].set_xlabel('paired gain vs corresponding ordinary arm');axes[1].set_title('Robust-estimation effect')
    fig.suptitle('Earlier simplified pipeline · superseded by the full Bayesian comparison',fontsize=14,y=1.02)
    fig.tight_layout();save(fig,'09_legacy_simplified_pipeline.png')


def invalid_synthetic_diagnostics():
    r=load('results/independent/synthetic_recovery.json')['diagnostics']
    arms=['ordinary','robust_2p5'];labels=['Ordinary','Robust 2.5'];x=np.arange(2)
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.8))
    axes[0].bar(x,[r[a]['failed_cells'] for a in arms],color=[C['ordinary'],C['robust']]);axes[0].set_ylim(0,17)
    axes[0].set_ylabel('failed cells (of 16)');axes[0].set_title('Cells failing diagnostics')
    axes[1].bar(x,[r[a]['max_rhat'] for a in arms],color=[C['ordinary'],C['robust']]);axes[1].axhline(1.01,color=C['negative'],ls='--')
    axes[1].set_ylim(1,2);axes[1].set_ylabel('maximum R-hat');axes[1].set_title('Worst R-hat')
    axes[2].bar(x,[r[a]['total_divergences'] for a in arms],color=[C['ordinary'],C['robust']]);axes[2].set_ylabel('divergences')
    axes[2].set_title('Total divergences')
    for ax in axes:ax.set_xticks(x,labels)
    fig.suptitle('Independent synthetic Bayesian recovery · invalid for inference',fontsize=14,y=1.03,color=C['negative'])
    fig.tight_layout();save(fig,'10_invalid_synthetic_diagnostics.png')


def early_real_tuning_events():
    e=load('results/early_real_2013/results.json')
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.3))
    styles=[('ordinary','Ordinary',C['ordinary']),('huber_2.5','Huber 2.5',C['robust']),
            ('block_16','Block-16',C['block'])]
    for arm,label,color in styles:
        rows=sorted([v for v in e['tuning'] if v['arm']==arm],key=lambda v:v['sigma'])
        axes[0].plot([v['sigma'] for v in rows],[v['point_logloss'] for v in rows],
                     marker='o',lw=1.8,color=color,label=label)
    axes[0].set_xscale('log');axes[0].set_xlabel('annual drift SD');axes[0].set_ylabel('2012 point log loss')
    axes[0].set_title('Drift tuning');axes[0].legend(frameon=False,fontsize=8.5)
    labels=['Train\n2011–12','Online eval\n2013'];x=np.arange(2)
    removed=[e['filtering']['training_2011_2012']['removed_points']/e['filtering']['training_2011_2012']['points'],
             e['filtering']['online_updates_2013']['removed_points']/e['filtering']['online_updates_2013']['points']]
    axes[1].bar(x,removed,color=[C['block'],'#36A9B2'],width=.58)
    axes[1].set_xticks(x,labels);axes[1].set_ylabel('fraction of observed points rejected')
    axes[1].set_title('Real-data block-filter activity')
    for i,v in enumerate(removed):axes[1].text(i,v+max(removed)*.035,f'{100*v:.3f}%',ha='center',fontsize=9)
    events=[('2013-580','Australian Open'),('2013-520','Roland Garros'),('2013-540','Wimbledon'),('2013-560','US Open')]
    width=.34;x=np.arange(4)
    for j,(arm,label,color) in enumerate(styles[1:]):
        ds=[e['summary']['frozen'][arm]['events'][key]['gain'] for key,_ in events]
        means=np.array([d['mean'] for d in ds]);lo=means-np.array([d['lo'] for d in ds]);hi=np.array([d['hi'] for d in ds])-means
        axes[2].errorbar(x+(j-.5)*width,means,yerr=[lo,hi],fmt='o',capsize=4,color=color,label=label)
    axes[2].axhline(0,color=C['ink'],ls='--',lw=1);axes[2].set_xticks(x,[v[1] for v in events],rotation=22,ha='right')
    axes[2].set_ylabel('frozen point-log-loss gain');axes[2].set_title('2013 event sensitivity');axes[2].legend(frameon=False,fontsize=8.5)
    fig.suptitle('Ordered real-point tuning and filter behavior',fontsize=14,y=1.02)
    fig.tight_layout();save(fig,'11_early_real_tuning_and_events.png')


def pilot_adaptation():
    r=load('results/point_robust_pilot/results.json')['summary']
    arms=[('ordinary','Ordinary',C['ordinary']),('huber_2.5','Huber',C['robust']),
          ('block_8','Block-8','#65B8BE'),('block_16','Block-16',C['block']),
          ('block_32','Block-32','#005F68')]
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharex=True)
    for ax,condition in zip(axes,['permanent_drop','permanent_rise']):
        labels=[];positions=[];y=0
        for gen in ('rw','smooth'):
            for arm,label,color in arms:
                item=next(v for v in r if v['generator']==gen and v['condition']==condition and v['arm']==arm)
                d=item['recovery']['lag_among_recovered'];errorbarh(ax,y,d,color)
                labels.append(f'{gen.upper()} · {label}');positions.append(y);y+=1
            y+=.5
        ax.set_yticks(positions,labels,fontsize=8);ax.invert_yaxis();ax.set_xlabel('sessions to sustained recovery')
        ax.set_title(condition.replace('_',' ').title())
    fig.suptitle('Synthetic permanent-change adaptation · all 40 focal trajectories recovered',fontsize=14,y=1.02)
    fig.text(.5,-.01,'Recovery requires serve and return errors ≤0.2 logits for three consecutive sessions.',ha='center',fontsize=9,color=C['muted'])
    fig.tight_layout();save(fig,'12_point_pilot_adaptation.png')


def index():
    text='''# Consolidated result figures

Generated from saved development artifacts; no model refits and no 2025 outcomes.

## Current evidence

![Executive summary](01_executive_summary.png)

![Full-model scores and calibration](02_full_model_scores_and_calibration.png)

![Full-model paired gains](03_full_model_paired_gains.png)

![Tuning and simulation](04_tuning_and_simulation.png)

![Sampling diagnostics](05_sampling_diagnostics.png)

![Early real point and match results](06_early_real_point_and_match.png)

![Synthetic point-pilot recovery](07_point_pilot_recovery.png)

![Synthetic point filtering](08_point_pilot_filtering.png)

## Provenance and failed checks

The older simplified online-filter results are retained for provenance and are
superseded by the full Bayesian comparison.

![Legacy simplified pipeline](09_legacy_simplified_pipeline.png)

The separate two-seed Bayesian synthetic recovery run failed its sampler
diagnostics. Its RMSE differences are deliberately not plotted as evidence.

![Invalid synthetic diagnostics](10_invalid_synthetic_diagnostics.png)

![Early real tuning and events](11_early_real_tuning_and_events.png)

![Synthetic permanent-change adaptation](12_point_pilot_adaptation.png)

Existing detailed figures remain available:

- [`point_robust_pilot/trajectories.png`](../../point_robust_pilot/trajectories.png)
- [`point_robust_pilot/block_weights.png`](../../point_robust_pilot/block_weights.png)
- [`early_real_2013/point_gains.png`](../../early_real_2013/point_gains.png)
- [`plots/skill_trajectories.png`](../skill_trajectories.png)
- [`plots/alpha_by_surface.png`](../alpha_by_surface.png)
- [`plots/tuning_grids.png`](../tuning_grids.png)
- [`plots/season_scores.png`](../season_scores.png)

Primary numerical sources: [`independent/comparison.json`](../../independent/comparison.json),
[`early_real_2013/results.json`](../../early_real_2013/results.json), and
[`point_robust_pilot/results.json`](../../point_robust_pilot/results.json).
'''
    (OUT/'INDEX.md').write_text(text)


def manifest():
    inputs=['results/independent/comparison.json','results/independent/comparison_2014_predictions.csv',
            'results/independent/early_real_calibration_control.json','results/independent/synthetic_recovery.json',
            'results/early_real_2013/results.json','results/point_robust_pilot/results.json',
            'results/robust_check/compare_2014.json','results/sim_forecast/sim_forecast_2014.json']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    data={'source_sha256':digest(Path(__file__)),
          'inputs':{p:digest(ROOT/p) for p in inputs},
          'figures':{p.name:digest(p) for p in sorted(OUT.glob('*.png'))},
          'note':'Generated from saved development artifacts; no refits and no 2025 outcomes.'}
    (OUT/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')


def main():
    executive_summary();full_scores_calibration();full_paired_gains();tuning_and_simulation()
    diagnostics();early_real();pilot_recovery();pilot_filtering();legacy_pipeline()
    invalid_synthetic_diagnostics();early_real_tuning_events();pilot_adaptation();index();manifest()
    print(f'wrote 12 figures, INDEX.md, and manifest.json to {OUT}')


if __name__=='__main__':main()
