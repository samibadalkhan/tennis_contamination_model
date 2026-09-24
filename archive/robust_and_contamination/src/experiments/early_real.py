"""Train on 2011–2012 point sequences; evaluate 2013 without retuning.

See docs/POINT_ROBUST_EARLY_REAL.md. Reads only explicitly allowed year files.
python -m src.experiments.early_real
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.special import expit

from src import trials
from src.independent.scoring import iid, rules
from src.models.ability import OnlineAbility
from src.models.block_filter import filter_points, blocks_for
from src.util import DATA, RESULTS, ROOT, ensure, write_json, utcnow

YEARS = (2011, 2012, 2013)
ARMS = ('ordinary', 'huber_2.5', 'block_16')
SIGMAS = (.03, .1, .3, 1.)
SLAMS = {'ausopen': 'Australian Open', 'frenchopen': 'Roland Garros',
         'wimbledon': 'Wimbledon', 'usopen': 'US Open'}
ROUND_ORDER = {'R128': 0, 'R64': 1, 'R32': 2, 'R16': 3, 'QF': 4, 'SF': 5, 'F': 6}
ALIASES = {'stanislaswawrinka': 'stanwawrinka', 'richardberankis': 'ricardasberankis',
           'rogeriodutradasilva': 'rogeriodutrasilva'}


def name_key(name):
    normalized = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    key = re.sub('[^a-z0-9]', '', normalized)
    return ALIASES.get(key, key)


@dataclass
class Match:
    match_id: str
    year: int
    tourney_id: str
    tournament: str
    date: pd.Timestamp
    round: str
    surface: str
    p1: int
    p2: int
    name1: str
    name2: str
    p1_won: bool
    retired: bool
    serving: np.ndarray
    outcome: np.ndarray
    point_numbers: np.ndarray


def verify_inputs(root=DATA, years=YEARS):
    if tuple(years) != YEARS:
        raise ValueError('this registered pilot only permits 2011, 2012 and 2013')
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    records = {r['file']: r['sha256'] for r in manifest['files']}
    paths = [root / 'atp' / f'atp_matches_{y}.csv' for y in years]
    paths += [root / 'slam_pointbypoint' / f'{y}-{s}-{kind}.csv'
              for y in years for s in SLAMS for kind in ('matches', 'points')]
    hashes = {}
    for path in paths:
        key = str(path.relative_to(root))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if records.get(key) != actual:
            raise ValueError(f'input SHA-256 mismatch: {key}')
        hashes[key] = actual
    return hashes


def load_early(root=DATA):
    hashes = verify_inputs(root)
    matches, audit = [], []
    for year in YEARS:
        atp = pd.read_csv(root / 'atp' / f'atp_matches_{year}.csv')
        for slam, tournament in SLAMS.items():
            meta = pd.read_csv(root / 'slam_pointbypoint' / f'{year}-{slam}-matches.csv')
            meta = meta[pd.to_numeric(meta.match_num).between(1000, 1999)]
            event = atp[(atp.tourney_level == 'G') & (atp.tourney_name == tournament)]
            lookup = {}
            for row in event.itertuples():
                key = frozenset((name_key(row.winner_name), name_key(row.loser_name)))
                if key in lookup:
                    raise ValueError(f'ambiguous ATP pair at {year}/{slam}: {key}')
                lookup[key] = row
            path = root / 'slam_pointbypoint' / f'{year}-{slam}-points.csv'
            points = pd.read_csv(path, usecols=['match_id', 'PointNumber', 'PointServer', 'PointWinner'])
            points = points[points.match_id.isin(meta.match_id)]
            played = points.PointServer.isin([1, 2]) & points.PointWinner.isin([1, 2])
            invalid = int((~played).sum())
            points = points[played]
            groups = {mid: g for mid, g in points.groupby('match_id', sort=False)}
            no_points, gaps, count_disagreements, label_disagreements = [], 0, [], []
            for row in meta.itertuples():
                key = frozenset((name_key(row.player1), name_key(row.player2)))
                if key not in lookup:
                    raise ValueError(f'unmatched point metadata: {row.match_id}, {row.player1}, {row.player2}')
                a = lookup[key]
                if row.match_id not in groups:
                    no_points.append(row.match_id)
                    continue
                g = groups[row.match_id]
                numbers = pd.to_numeric(g.PointNumber, errors='raise').to_numpy(dtype=float)
                blocks_for(numbers, 16)  # fail on duplicates/non-chronological numbers
                gaps += int(np.sum(np.diff(numbers) != 1))
                first_is_winner = name_key(row.player1) == name_key(a.winner_name)
                p1, p2 = (a.winner_id, a.loser_id) if first_is_winner else (a.loser_id, a.winner_id)
                name1, name2 = (a.winner_name, a.loser_name) if first_is_winner else (a.loser_name, a.winner_name)
                if pd.notna(row.winner) and int(row.winner) != (1 if first_is_winner else 2):
                    label_disagreements.append(row.match_id)
                if a.round not in ROUND_ORDER or int(a.best_of) != 5:
                    raise ValueError(f'unsupported round or format: {row.match_id}')
                serving = g.PointServer.to_numpy(dtype=int) - 1
                outcome = (g.PointServer == g.PointWinner).to_numpy(dtype=int)
                totals = np.array([(serving == s).sum() for s in (0, 1)])
                stats = np.array([a.w_svpt, a.l_svpt] if first_is_winner else [a.l_svpt, a.w_svpt])
                if np.isfinite(stats).all() and not np.array_equal(totals, stats):
                    count_disagreements.append({'match_id': row.match_id, 'recorded': totals.tolist(),
                                                'atp': stats.astype(int).tolist()})
                matches.append(Match(row.match_id, year, str(a.tourney_id), tournament,
                                     pd.to_datetime(str(a.tourney_date), format='%Y%m%d'),
                                     a.round, a.surface, int(p1), int(p2), name1, name2,
                                     first_is_winner, 'RET' in str(a.score).upper(),
                                     serving, outcome, numbers.astype(int)))
            audit.append({'year': year, 'slam': slam, 'atp_matches': len(event),
                          'point_metadata_matches': len(meta), 'played_matches': len(groups),
                          'played_points': len(points), 'invalid_or_sentinel_rows': invalid,
                          'no_played_points': no_points, 'point_number_gaps': gaps,
                          'serve_count_disagreements': count_disagreements,
                          'winner_label_disagreements': label_disagreements})
    matches.sort(key=lambda m: (m.date, m.tourney_id, ROUND_ORDER[m.round], m.match_id))
    if len({m.match_id for m in matches}) != len(matches):
        raise ValueError('duplicate match IDs')
    # Unknown within-round dates cannot leak via another encounter involving
    # the same player: require a true single-elimination round partition.
    seen = {}
    for m in matches:
        key = (m.tourney_id, m.round)
        players = seen.setdefault(key, set())
        if players.intersection((m.p1, m.p2)):
            raise ValueError(f'player repeats within round: {key}')
        players.update((m.p1, m.p2))
    return matches, {'input_sha256': hashes, 'coverage': audit, 'aliases': ALIASES}


def surface_intercepts(matches):
    totals = {}
    for m in matches:
        pair = totals.setdefault(m.surface, np.zeros(2))
        pair += [m.outcome.sum(), len(m.outcome)]
    return {s: float(np.log((k + .5) / (n - k + .5))) for s, (k, n) in totals.items()}


def make_filter(alpha, sigma, arm):
    return OnlineAbility(alpha, sigma=sigma, prior_var=1., robust_c=2.5 if arm == 'huber_2.5' else None)


def predict(f, m):
    # Read-only: a player absent at the training cutoff gets prior mean zero.
    return tuple(float(expit(f._alpha(m.surface) + f.sm.get(a, 0.) - f.rm.get(b, 0.)))
                 for a, b in ((m.p1, m.p2), (m.p2, m.p1)))


def learn(f, m, arm, cache):
    keep = np.ones(len(m.outcome), bool)
    if arm == 'block_16':
        if m.match_id not in cache:
            cache[m.match_id] = filter_points(m.serving, m.outcome, m.point_numbers)
        keep = cache[m.match_id][0]
    for player in (m.p1, m.p2):
        f._ensure(player, m.date)
        if m.date < f.last[player]:
            raise ValueError('non-chronological ability update')
        f._diffuse(player, m.date)
    for side, server, returner in ((0, m.p1, m.p2), (1, m.p2, m.p1)):
        use = keep & (m.serving == side)
        n = int(use.sum())
        if n:
            f._update_serve_obs(server, returner, int(m.outcome[use].sum()), n, m.surface)


def point_rows(m, probabilities, seen):
    rows = []
    for side, player, opponent in ((0, m.p1, m.p2), (1, m.p2, m.p1)):
        use = m.serving == side
        n, k = int(use.sum()), int(m.outcome[use].sum())
        if not n:
            continue
        p = float(np.clip(probabilities[side], 1e-9, 1-1e-9))
        rows.append({'match_id': m.match_id, 'tourney_id': m.tourney_id,
                     'server': player, 'opponent': opponent, 'surface': m.surface,
                     'round': m.round, 'retired': m.retired,
                     'both_seen': m.p1 in seen and m.p2 in seen,
                     'p': p, 'n': n, 'k': k,
                     'loss_sum': -k * np.log(p) - (n-k) * np.log1p(-p),
                     'brier_sum': k * (1-p)**2 + (n-k) * p**2,
                     'correct': k if p > .5 else n-k})
    return rows


def evaluate(f, matches, arm, mode, seen, cache):
    if mode not in ('frozen', 'online'):
        raise ValueError(mode)
    points, outcomes = [], []
    for m in matches:
        pa, pb = predict(f, m)  # ALWAYS before consulting this match's outcomes
        points.extend(point_rows(m, (pa, pb), seen))
        final_at, final_to = rules(m.tournament, m.year, 5)
        p = float(np.clip(iid(pa, pb, 5, final_at, final_to), 1e-9, 1-1e-9))
        y = int(m.p1_won)
        outcomes.append({'match_id': m.match_id, 'tourney_id': m.tourney_id,
                         'p1': m.p1, 'p2': m.p2, 'p': p, 'y': y, 'round': m.round,
                         'retired': m.retired, 'both_seen': m.p1 in seen and m.p2 in seen,
                         'n': 1, 'loss_sum': -y*np.log(p)-(1-y)*np.log1p(-p),
                         'correct': int((p > .5) == y)})
        if mode == 'online':
            learn(f, m, arm, cache)
    return pd.DataFrame(points).assign(arm=arm, mode=mode), pd.DataFrame(outcomes).assign(arm=arm, mode=mode)


def clustered_interval(frame, value, clusters):
    if frame.empty:
        return None
    agg = frame.groupby(clusters, sort=True)[[value, 'n']].sum().to_numpy(float)
    if not len(agg) or agg[:, 1].sum() <= 0:
        raise ValueError('empty bootstrap clusters')
    rng = np.random.default_rng(913)
    sums = agg[rng.integers(len(agg), size=(2000, len(agg)))].sum(axis=1)
    boot = sums[:, 0] / sums[:, 1]
    return {'mean': float(agg[:, 0].sum()/agg[:, 1].sum()),
            'lo': float(np.quantile(boot, .025)), 'hi': float(np.quantile(boot, .975)),
            'clusters': len(agg)}


def paired_gain(base, candidate, clusters, keys):
    joined = base[keys + ['n', 'loss_sum']].merge(candidate[keys + ['n', 'loss_sum']],
                                                on=keys, validate='one_to_one', suffixes=('_base', '_arm'))
    if len(joined) != len(base) or len(joined) != len(candidate) or not np.array_equal(joined.n_base, joined.n_arm):
        raise ValueError('paired scoring rows or denominators differ')
    joined['gain'] = joined.loss_sum_base - joined.loss_sum_arm
    joined['n'] = joined.n_base
    return clustered_interval(joined, 'gain', clusters)


def summarize(points, matches):
    result = {}
    pkeys = ['match_id', 'tourney_id', 'server']
    mkeys = ['match_id', 'tourney_id']
    for mode in ('frozen', 'online'):
        ps, ms = points[points['mode'] == mode], matches[matches['mode'] == mode]
        basep, basem = ps[ps.arm == 'ordinary'], ms[ms.arm == 'ordinary']
        result[mode] = {}
        for arm in ARMS:
            p, m = ps[ps.arm == arm], ms[ms.arm == arm]
            d = {'point_logloss': clustered_interval(p, 'loss_sum', ['tourney_id', 'server']),
                 'point_brier': clustered_interval(p, 'brier_sum', ['tourney_id', 'server']),
                 'point_accuracy': clustered_interval(p, 'correct', ['tourney_id', 'server']),
                 'point_gain': paired_gain(basep, p, ['tourney_id', 'server'], pkeys),
                 'point_gain_tournament': paired_gain(basep, p, ['tourney_id'], pkeys),
                 'match_logloss': clustered_interval(m, 'loss_sum', ['tourney_id']),
                 'match_accuracy': clustered_interval(m, 'correct', ['tourney_id']),
                 'match_gain': paired_gain(basem, m, ['tourney_id'], mkeys),
                 'completed_match_logloss': clustered_interval(m[~m.retired], 'loss_sum', ['tourney_id']),
                 'completed_match_gain': paired_gain(basem[~basem.retired], m[~m.retired], ['tourney_id'], mkeys),
                 'subsets': {}, 'events': {}}
            for label, predicate in [('both_seen', lambda x: x.both_seen),
                                     ('opening_round', lambda x: x['round'] == 'R128'),
                                     ('later_round', lambda x: x['round'] != 'R128')]:
                b, a = basep[predicate(basep)], p[predicate(p)]
                d['subsets'][label] = {'points': int(a.n.sum()), 'matches': int(a.match_id.nunique()),
                                       'logloss': clustered_interval(a, 'loss_sum', ['tourney_id', 'server']),
                                       'gain': paired_gain(b, a, ['tourney_id', 'server'], pkeys)}
            for event, a in p.groupby('tourney_id'):
                b = basep[basep.tourney_id == event]
                d['events'][str(event)] = {'points': int(a.n.sum()),
                                          'gain': paired_gain(b, a, ['tourney_id', 'server'], pkeys)}
            result[mode][arm] = d
    return result


def fmt(d, digits=5):
    return '—' if d is None else f"{d['mean']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def report(result, out):
    lines = ['# Early real-data comparison', '',
             '**Train: 2011–2012. Evaluate: 2013.** Men’s Grand Slam matches with recorded points. '
             'Drift was selected on 2012 only. This is a development read, not the untouched 2025 test.', '',
             f"Training: {result['training']['matches']} matches, {result['training']['points']:,} points. "
             f"Evaluation: {result['evaluation']['matches']} matches, {result['evaluation']['points']:,} points. "
             f"Training players: {result['training']['players']}; unseen evaluation players: {result['evaluation']['unseen_players']}.", '',
             'All intervals are 95% clustered bootstrap intervals. Positive paired gain means lower '
             'loss than ordinary. Point clusters are server-player/tournament; match intervals '
             'use tournaments, with only four clusters. Neither interval fully resolves all '
             'cross-player and cross-event dependencies.', '',
             '## Frozen end-of-2012 abilities (primary)', '',
             'No 2013 outcomes update these abilities. Every held-out point is scored, including '
             'points a robust updater would reject.', '',
             '| Arm | Selected annual drift SD | Point log loss [CI] | Paired point gain [CI] | Match log loss [CI] | Match accuracy [CI] |',
             '|---|---|---|---|---|---|']
    for arm, d in result['summary']['frozen'].items():
        lines.append(f"| {arm} | {result['chosen_sigma'][arm]} | {fmt(d['point_logloss'])} | "
                     f"{fmt(d['point_gain'])} | {fmt(d['match_logloss'], 4)} | {fmt(d['match_accuracy'], 3)} |")
    lines += ['', '## Updating after each 2013 match (secondary)', '',
              'Predict first, update afterward. Hyperparameters remain frozen.', '',
              '| Arm | Point log loss [CI] | Paired point gain [CI] | Match log loss [CI] | Paired match gain [CI] |',
              '|---|---|---|---|---|']
    for arm, d in result['summary']['online'].items():
        lines.append(f"| {arm} | {fmt(d['point_logloss'])} | {fmt(d['point_gain'])} | "
                     f"{fmt(d['match_logloss'], 4)} | {fmt(d['match_gain'], 4)} |")
    lines += ['', '## Point-loss sensitivity checks', '',
              '| Mode | Arm | Tournament-cluster paired gain [CI] | Both players seen before 2013 [CI] | Opening round gain [CI] | Later-round gain [CI] |',
              '|---|---|---|---|---|---|']
    for mode, arms in result['summary'].items():
        for arm, d in arms.items():
            if arm != 'ordinary':
                lines.append(f"| {mode} | {arm} | {fmt(d['point_gain_tournament'])} | "
                             f"{fmt(d['subsets']['both_seen']['gain'])} | "
                             f"{fmt(d['subsets']['opening_round']['gain'])} | {fmt(d['subsets']['later_round']['gain'])} |")
    lines += ['', '## Completed-match sensitivity', '',
              'Retirements remain in point fitting and the headline scores. The following removes '
              'retirements only from match-outcome scoring, since the iid forecast has no retirement model.', '',
              '| Mode | Arm | Completed-match log loss [CI] | Paired gain [CI] |',
              '|---|---|---|---|']
    for mode, arms in result['summary'].items():
        for arm, d in arms.items():
            lines.append(f"| {mode} | {arm} | {fmt(d['completed_match_logloss'], 4)} | {fmt(d['completed_match_gain'], 4)} |")
    lines += ['', '## Filtering activity', '']
    for key, d in result['filtering'].items():
        lines.append(f"- {key}: rejected {d['removed_points']:,}/{d['points']:,} observed points "
                     f"in {d['affected_matches']}/{d['matches']} matches. These are filter decisions, not confirmed contamination.")
    lines += ['', '## Data and interpretation limits', '',
              'Point coverage is incomplete and favors recorded courts. These are point-corpus-only '
              'fits, not models trained on every ATP match. ATP records supply stable identities '
              'and match metadata; they are not extra ability observations. Full-name aliases '
              'and per-file SHA-256 checks are recorded in audit.json.', '',
              'Matches follow verified bracket rounds. Tournament start dates govern diffusion '
              'because exact match dates are absent; there is no within-event diffusion. Missing-point '
              'gaps split blocks. Recorded point counts may differ from ATP serve totals; the audit '
              'preserves those discrepancies, and no outcomes are imputed.', '',
              'The block filter now conditions on actual unequal service opportunities and uses a '
              'point-count rejection budget. The Huber and ordinary arms share its observation '
              'cohort and underlying approximate ability model. A better held-out forecast would '
              'not by itself prove that temporary impairment was removed. Conversely, a small '
              'or negative forecasting effect does not identify the true latent skill error.', '',
              'Match forecasts are the same iid scoring recursion in all arms with historical '
              'deciding-set rules. This isolates estimation, rather than testing the final '
              'per-arm-tuned burst simulator. See [protocol](../../docs/POINT_ROBUST_EARLY_REAL.md).', '',
              '![Paired held-out point gains](point_gains.png)', '',
              'Full paired predictions: `point_predictions.csv`, `match_predictions.csv`. '
              'Training snapshots: `abilities_2012.csv`. Filters: `block_diagnostics.json`. '
              'Tuning, intervals and source hashes: `results.json`.', '']
    (out / 'report.md').write_text('\n'.join(lines))


def plot(result, out):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tennis_point_robust_matplotlib')
    os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/tennis_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, ('frozen', 'online')):
        for y, arm in enumerate(('huber_2.5', 'block_16')):
            d = result['summary'][mode][arm]['point_gain']
            ax.errorbar(d['mean'], y, xerr=[[d['mean']-d['lo']], [d['hi']-d['mean']]],
                        fmt='o', capsize=5, color='#007f8b' if arm == 'block_16' else '#a585bd')
        ax.axvline(0, color='#777777', linestyle='--', linewidth=1)
        ax.set_yticks([0, 1], ['Huber 2.5', 'Block filter (16)'])
        ax.set_ylim(-.5, 1.5)
        ax.set_title('Frozen after 2012' if mode == 'frozen' else 'Update after each 2013 match')
        ax.set_xlabel('Point log-loss reduction vs ordinary\nPositive = better; 95% player/event bootstrap')
        ax.grid(axis='x', alpha=.15)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    fig.suptitle('2013 real points · train through 2012 · all held-out points scored')
    fig.savefig(out / 'point_gains.png', dpi=170)
    plt.close(fig)


def run(out):
    if out.exists():
        raise FileExistsError(f'{out} exists; choose a new --out directory')
    ensure(out)
    protocol = {'train_years': [2011, 2012], 'tune_year': 2012, 'eval_year': 2013,
                'arms': ARMS, 'sigma_grid': SIGMAS, 'prior_var': 1., 'block_size': 16,
                'null_quantile': .99, 'null_draws': 255, 'max_removed_point_fraction': .25}
    source_paths = ['src/experiments/early_real.py', 'src/models/block_filter.py',
                    'src/models/ability.py', 'src/independent/scoring.py',
                    'docs/POINT_ROBUST_EARLY_REAL.md']
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    write_json(out / 'config.json', {'utc': utcnow(), 'protocol': protocol, 'source_sha256': hashes})
    matches, audit = load_early()
    write_json(out / 'audit.json', audit)
    train = [m for m in matches if m.year <= 2012]
    test = [m for m in matches if m.year == 2013]
    warmup = [m for m in train if m.year == 2011]
    validation = [m for m in train if m.year == 2012]
    print(f'Loaded {len(train)} training and {len(test)} evaluation matches.', flush=True)
    alpha_tune = surface_intercepts(warmup)
    alpha_final = surface_intercepts(train)
    cache, tuning, chosen = {}, [], {}
    for arm in ARMS:
        candidates = []
        for sigma in SIGMAS:
            trial = trials.log(stage='early-real-points', kind='tune/' + arm,
                               params={**protocol, 'sigma': sigma, 'source_sha256': hashes},
                               note='2011 initialization; 2012 pre-match validation only; started.')
            f = make_filter(alpha_tune, sigma, arm)
            for m in warmup:
                learn(f, m, arm, cache)
            seen_warmup = set(f.sm)
            predictions, _ = evaluate(f, validation, arm, 'online', seen_warmup, cache)
            loss = float(predictions.loss_sum.sum() / predictions.n.sum())
            d = {'arm': arm, 'sigma': sigma, 'point_logloss': loss, 'trial_id': trial}
            candidates.append(d)
            tuning.append(d)
            write_json(out / 'tuning.json', tuning)
            print(f'Tune {arm} sigma={sigma}: 2012 point loss {loss:.6f}', flush=True)
        chosen[arm] = min(candidates, key=lambda d: d['point_logloss'])['sigma']
    write_json(out / 'selection_before_2013.json', {'chosen_sigma': chosen,
                                                   'alpha_tune': alpha_tune, 'alpha_final': alpha_final})
    allpoints, allmatches, snapshots, final_trials = [], [], [], []
    names = {pid: name for m in train for pid, name in ((m.p1, m.name1), (m.p2, m.name2))}
    seen = set(names)
    for arm in ARMS:
        f = make_filter(alpha_final, chosen[arm], arm)
        for m in train:
            learn(f, m, arm, cache)
        for player in sorted(seen):
            snapshots.append({'arm': arm, 'player_id': player, 'name': names[player],
                              'serve': f.sm[player], 'return': f.rm[player],
                              'serve_variance': f.sv[player], 'return_variance': f.rv[player]})
        for mode in ('frozen', 'online'):
            tid = trials.log(stage='early-real-points', kind=f'2013/{mode}/{arm}',
                             params={**protocol, 'sigma': chosen[arm], 'source_sha256': hashes},
                             note='Frozen selection; all evaluation points scored; started.')
            final_trials.append({'arm': arm, 'mode': mode, 'trial_id': tid})
            p, m = evaluate(copy.deepcopy(f), test, arm, mode, seen, cache)
            allpoints.append(p)
            allmatches.append(m)
    points, outcomes = pd.concat(allpoints, ignore_index=True), pd.concat(allmatches, ignore_index=True)
    points.to_csv(out / 'point_predictions.csv', index=False)
    outcomes.to_csv(out / 'match_predictions.csv', index=False)
    pd.DataFrame(snapshots).to_csv(out / 'abilities_2012.csv', index=False)
    filtering, diagnostics = {}, []
    for label, subset in (('training_2011_2012', train), ('online_updates_2013', test)):
        removed = 0
        affected = 0
        for m in subset:
            keep, info = cache[m.match_id]
            removed += int((~keep).sum())
            affected += int((~keep).any())
            diagnostics.append({'match_id': m.match_id, 'year': m.year, 'tourney_id': m.tourney_id,
                                'p1': m.p1, 'p2': m.p2, 'name1': m.name1, 'name2': m.name2,
                                'retired': m.retired, 'points': len(keep),
                                'removed_point_numbers': m.point_numbers[~keep].tolist(), **info})
        filtering[label] = {'matches': len(subset), 'points': sum(len(m.outcome) for m in subset),
                            'removed_points': removed, 'affected_matches': affected}
    write_json(out / 'block_diagnostics.json', diagnostics)
    result = {'utc': utcnow(), 'protocol': protocol, 'source_sha256': hashes,
              'input_sha256': audit['input_sha256'], 'tuning': tuning,
              'chosen_sigma': chosen, 'alpha_tune': alpha_tune, 'alpha_final': alpha_final,
              'final_trials': final_trials,
              'training': {'matches': len(train), 'points': sum(len(m.outcome) for m in train), 'players': len(seen)},
              'evaluation': {'matches': len(test), 'points': sum(len(m.outcome) for m in test),
                             'unseen_players': len({p for m in test for p in (m.p1, m.p2)} - seen)},
              'filtering': filtering, 'summary': summarize(points, outcomes)}
    write_json(out / 'results.json', result)
    plot(result, out)
    report(result, out)
    print(f'Report: {out / "report.md"}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS / 'early_real_2013')
    run(parser.parse_args().out)
