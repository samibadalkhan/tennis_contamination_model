"""Independent exact iid recursion and coherent non-iid point simulator.

One optional 40-point impairment starts uniformly in points 0..80, affecting
both serve and return. Outcomes, serve totals, and winner share one realization.
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def game(p):
    q=1-p
    return p**4*(1+4*q+10*q*q)+20*p**3*q**3*p*p/(p*p+q*q)


@njit(cache=True)
def tiebreak(pa,pb,to=7):
    d=np.zeros((to+1,to+1));d[0,0]=1.
    win=0.
    for total in range(2*to-1):
        for a in range(to+1):
            b=total-a
            if b<0 or b>to:continue
            mass=d[a,b]
            if a==to:
                win+=mass;continue
            if b==to:continue
            if a==to-1 and b==to-1:
                win+=mass*pa*(1-pb)/(pa*(1-pb)+(1-pa)*pb)
                continue
            p=pa if ((total+1)//2)%2==0 else 1-pb
            d[a+1,b]+=mass*p;d[a,b+1]+=mass*(1-p)
    return win


@njit(cache=True)
def setprob(pa,pb,at=6,to=7):
    ha=game(pa);hb=game(pb)
    limit=max(6,at)
    d=np.zeros((limit+2,limit+2));d[0,0]=1.
    win=0.
    for total in range(2*limit+1):
        for a in range(limit+2):
            b=total-a
            if b<0 or b>=limit+2:continue
            mass=d[a,b]
            if max(a,b)>=6 and abs(a-b)>=2:
                if a>b:win+=mass
                continue
            if a==b and a==limit:
                if at==0:
                    win+=mass*ha*(1-hb)/(ha*(1-hb)+(1-ha)*hb)
                else:
                    win+=mass*tiebreak(pa,pb,to)
                continue
            p=ha if total%2==0 else 1-hb
            d[a+1,b]+=mass*p;d[a,b+1]+=mass*(1-p)
    return win


@njit(cache=True)
def iid(pa,pb,best_of=3,final_at=6,final_to=7):
    p=setprob(pa,pb);q=setprob(pa,pb,final_at,final_to)
    if best_of==3:return p*p+2*p*(1-p)*q
    return p**3*(1+3*(1-p))+6*p*p*(1-p)**2*q


def rules(tournament,year,best_of):
    if tournament in ['Australian Open','FO - RG','Roland Garros','Wimbledon','US Open']:
        if year>=2022:return 6,10
        if tournament=='Australian Open':return (6,10) if year>=2019 else (0,7)
        if tournament=='Wimbledon':return (12,7) if year>=2019 else (0,7)
        if tournament in ['FO - RG','Roland Garros']:return 0,7
    return 6,7


@njit(cache=True)
def play(pa,pb,best_of,final_at,final_to,start_a,end_a,sev_a,start_b,end_b,sev_b):
    """Uses current RNG. Returns winner, serve wins/counts and point count."""
    need=best_of//2+1
    sa=sb=ga=gb=a=b=0
    na=nb=ka=kb=0
    server=np.random.randint(2);tb_first=0;tb=False;tb_idx=0
    la=math.log(pa/(1-pa));lb=math.log(pb/(1-pb))
    for point in range(100000):
        ca=sev_a if start_a<=point<end_a else 0.
        cb=sev_b if start_b<=point<end_b else 0.
        effective=(tb_first ^ (((tb_idx+1)//2)%2)) if tb else server
        logit=(la-ca+cb) if effective==0 else (lb-cb+ca)
        serve_won=np.random.random()<1/(1+math.exp(-logit))
        if effective==0:
            na+=1;ka+=int(serve_won)
        else:
            nb+=1;kb+=int(serve_won)
        a_won=serve_won if effective==0 else not serve_won
        a+=int(a_won);b+=int(not a_won)
        deciding=(sa==need-1 and sb==need-1)
        if tb:
            tb_idx+=1
            target=final_to if deciding else 7
            if max(a,b)>=target and abs(a-b)>=2:
                sa+=int(a>b);sb+=int(b>a)
                server=1-tb_first;tb=False;ga=gb=a=b=0
        elif max(a,b)>=4 and abs(a-b)>=2:
            ga+=int(a>b);gb+=int(b>a);a=b=0;server=1-server
            if max(ga,gb)>=6 and abs(ga-gb)>=2:
                sa+=int(ga>gb);sb+=int(gb>ga);ga=gb=0
            else:
                at=final_at if deciding else 6
                if at>0 and ga==at and gb==at:
                    tb=True;tb_first=server;tb_idx=0
        if sa==need or sb==need:
            return int(sa>sb),ka,na,kb,nb,point+1
    raise RuntimeError('Unfinished simulated match; never silently truncate')


@njit(cache=True,parallel=True)
def posterior_iid(pa,pb,best_of,final_at,final_to):
    out=np.zeros(len(pa))
    for i in prange(len(pa)):
        for j in range(pa.shape[1]):
            out[i]+=iid(pa[i,j],pb[i,j],best_of[i],final_at[i],final_to[i])
        out[i]/=pa.shape[1]
    return out


def _game_vector(p):
    q=1-p
    return p**4*(1+4*q+10*q*q)+20*p**3*q**3*p*p/(p*p+q*q)


def _tiebreak_vector(pa,pb,to):
    """Exact tiebreak probability, vectorized over paired posterior draws."""
    n=len(pa);dp=np.zeros((to,to,n));dp[0,0]=1.;win=np.zeros(n)
    for total in range(2*to-1):
        for a in range(to):
            b=total-a
            if b<0 or b>=to:continue
            mass=dp[a,b]
            if a==to-1 and b==to-1:
                w=pa*(1-pb);l=(1-pa)*pb
                win+=mass*w/(w+l)
                continue
            p=pa if ((total+1)//2)%2==0 else 1-pb
            if a+1>=to and a+1-b>=2:win+=mass*p
            elif a+1<to:dp[a+1,b]+=mass*p
            if not (b+1>=to and b+1-a>=2):
                if b+1<to:dp[a,b+1]+=mass*(1-p)
    return win


def _set_vector(pa,pb,at,to):
    """Exact set probability using the repeating two-game cycle after 5-5."""
    ha=_game_vector(pa);hb=_game_vector(pb);n=len(pa)
    dp=np.zeros((6,6,n));dp[0,0]=1.;win=np.zeros(n)
    for total in range(11):
        for a in range(6):
            b=total-a
            if b<0 or b>5:continue
            mass=dp[a,b]
            if a==5 and b==5:
                w=ha*(1-hb);l=(1-ha)*hb
                r=1-w-l
                if at==0:tail=w/(w+l)
                else:
                    cycles=int(at)-5
                    tail=w*(1-r**cycles)/(1-r)+r**cycles*_tiebreak_vector(pa,pb,to)
                win+=mass*tail
                continue
            p=ha if total%2==0 else 1-hb
            if a+1>=6 and a+1-b>=2:win+=mass*p
            elif a+1<=5:dp[a+1,b]+=mass*p
            if not (b+1>=6 and b+1-a>=2):
                if b+1<=5:dp[a,b+1]+=mass*(1-p)
    return win


def iid_vector(pa,pb,best_of=3,final_at=6,final_to=7):
    p=_set_vector(np.asarray(pa),np.asarray(pb),6,7)
    q=p if (final_at==6 and final_to==7) else _set_vector(np.asarray(pa),np.asarray(pb),final_at,final_to)
    if best_of==3:return p*p+2*p*(1-p)*q
    return p**3+3*p**3*(1-p)+6*p*p*(1-p)**2*q


def posterior_iid_vectorized(pa,pb,best_of,final_at,final_to,batch_size=16):
    """Exact posterior mean, batched over matches and draws without approximation."""
    out=np.empty(len(pa));draws=pa.shape[1]
    formats=sorted(set(zip(best_of.tolist(),final_at.tolist(),final_to.tolist())))
    for bo,at,to in formats:
        ids=np.flatnonzero((best_of==bo)&(final_at==at)&(final_to==to))
        for start in range(0,len(ids),batch_size):
            ix=ids[start:start+batch_size]
            values=iid_vector(pa[ix].reshape(-1),pb[ix].reshape(-1),int(bo),int(at),int(to))
            out[ix]=values.reshape(len(ix),draws).mean(axis=1)
    return out


@njit(cache=True,parallel=True)
def posterior_burst(pa,pb,best_of,final_at,final_to,simulations=16384,seed=4001,severity=1.5,length=40):
    """Conditional on a potential burst; onset can occur after the match ends.

    Samples paired posterior draws. Identical per-match seeds couple estimators.
    Rate mixing with exact iid results occurs outside this function.
    """
    out=np.zeros(len(pa))
    for i in prange(len(pa)):
        np.random.seed(seed+104729*i)
        for j in range(simulations):
            k=np.random.randint(pa.shape[1])
            start=np.random.randint(81);who=np.random.randint(2)
            sa=severity if who==0 else 0.
            sb=severity if who==1 else 0.
            outcome=play(pa[i,k],pb[i,k],best_of[i],final_at[i],final_to[i],
                         start,start+length,sa,start,start+length,sb)
            out[i]+=outcome[0]
        # Jeffreys smoothing only for finite-simulation component.
        out[i]=(out[i]+0.5)/(simulations+1)
    return out
