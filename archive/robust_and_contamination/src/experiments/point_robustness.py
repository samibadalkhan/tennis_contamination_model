"""Ordered-point mechanism pilot. See docs/POINT_ROBUST_PILOT.md.

Run with python -m src.experiments.point_robustness. No real data is read.
The spectral prefilter is experimental; it is not published SEVER/MMW.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from scipy.special import expit

from src import trials
from src.models.ability import OnlineAbility
from src.util import RESULTS, ROOT, ensure, utcnow, write_json

ALPHA = float(np.log(.64 / .36))
BLOCK_SIZES = (8, 16, 32)
ARMS = ('ordinary', 'huber_2.5', 'block_8', 'block_16', 'block_32', 'oracle')
CONDITIONS = ('clean', 'burst', 'permanent_drop', 'permanent_rise')
GENERATORS = ('rw', 'smooth')
MAX_REMOVAL = .25
NULL_DRAWS = 255
NULL_QUANTILE = .99


@dataclass(frozen=True)
class Config:
    players: int = 8
    sessions: int = 64
    points: int = 192
    onset: int = 24
    burst_end: int = 36
    burst_points: int = 48
    severity: float = 2.
    jump: float = .6
    drift: float = .04
    prior_var: float = 1.
    seeds: int = 20


@dataclass(frozen=True)
class Observation:
    session: int
    a: int
    b: int
    serving: np.ndarray  # 0 = a, 1 = b
    outcome: np.ndarray  # 1 = server wins


def world(cfg: Config, generator: str, condition: str, seed: int):
    """Truth and corruption labels travel separately from observations."""
    if generator not in GENERATORS or condition not in CONDITIONS:
        raise ValueError('unknown generator or condition')
    rng = np.random.default_rng(17000 + seed)
    initial = rng.normal(0, .35, (cfg.players, 2))
    initial -= initial.mean()  # one common unidentified offset
    if generator == 'rw':
        increments = rng.normal(0, cfg.drift, (cfg.sessions, cfg.players, 2))
        increments[0] = 0
        truth = initial + np.cumsum(increments, axis=0)
    else:
        phase = rng.uniform(0, 2 * np.pi, (cfg.players, 2))
        t = np.arange(cfg.sessions)[:, None, None]
        truth = initial + .22 * (np.sin(phase + t / 18) - np.sin(phase))
    pairs = np.array([rng.permutation(cfg.players).reshape(-1, 2)
                      for _ in range(cfg.sessions)])
    uniforms = rng.random((cfg.sessions, cfg.players // 2, cfg.points))
    starts = rng.integers(0, cfg.points - cfg.burst_points + 1,
                          (cfg.sessions, cfg.players))
    if condition.startswith('permanent'):
        truth[cfg.onset:, :2] += cfg.jump * (-1 if condition.endswith('drop') else 1)
    serving = np.tile(np.repeat([0, 1], 4), cfg.points // 8)
    observations, masks = [], []
    index = np.arange(cfg.points)
    for t, encounters in enumerate(pairs):
        for j, (a, b) in enumerate(encounters):
            shift_a = np.zeros(cfg.points)
            shift_b = np.zeros(cfg.points)
            if condition == 'burst' and cfg.onset <= t < cfg.burst_end:
                for player, shift in ((a, shift_a), (b, shift_b)):
                    if player < 2:
                        affected = (index >= starts[t, player]) & (
                            index < starts[t, player] + cfg.burst_points)
                        shift[affected] = -cfg.severity
            # Impairment lowers both serve and return skill: the opponent's
            # serve probability rises when the impaired player returns.
            contrast_shift = np.where(serving == 0, shift_a - shift_b,
                                       shift_b - shift_a)
            logits = np.where(serving == 0,
                              ALPHA + truth[t, a, 0] - truth[t, b, 1],
                              ALPHA + truth[t, b, 0] - truth[t, a, 1])
            outcomes = (uniforms[t, j] < expit(logits + contrast_shift)).astype(int)
            observations.append(Observation(t, int(a), int(b), serving.copy(), outcomes))
            # Equal simultaneous impairments cancel in this specified model.
            masks.append(contrast_shift != 0)
    return truth, observations, masks


def block_counts(obs: Observation, block_size: int) -> tuple[np.ndarray, int]:
    if block_size % 8 or len(obs.outcome) % block_size:
        raise ValueError('blocks must contain complete eight-point service cycles')
    shape = (-1, block_size)
    y = obs.outcome.reshape(shape)
    serving = obs.serving.reshape(shape)
    if not np.all((serving == 0).sum(axis=1) == block_size // 2):
        raise ValueError('pilot requires balanced service opportunities per block')
    return np.stack([(y * (serving == s)).sum(axis=1) for s in (0, 1)], axis=1), block_size // 2


def standardized_scores(counts: np.ndarray, n: int) -> np.ndarray:
    # Half-count regularization is numerical only; centering removes its
    # constant score offset. It also defines the null's variance convention.
    p = (counts.sum(axis=0) + .5) / (n * len(counts) + 1)
    scores = (counts - n * p) / np.sqrt(n * p * (1 - p))
    return scores - scores.mean(axis=0)


@lru_cache(maxsize=16000)
def _conditional_threshold(blocks: int, n: int, k0: int, k1: int) -> float:
    """Condition on service totals: randomize their allocation across blocks.

    This uses observed totals only, never simulation truth. The fixed null
    seed makes calibration independent of fit order and execution caching.
    """
    rng = np.random.default_rng(np.random.SeedSequence([7193, blocks, n, k0, k1]))
    colors = np.full(blocks, n)
    samples = np.stack([rng.multivariate_hypergeometric(colors, k, size=NULL_DRAWS)
                        for k in (k0, k1)], axis=2)
    p = (np.array([k0, k1]) + .5) / (blocks * n + 1)
    g = (samples - samples.mean(axis=1, keepdims=True)) / np.sqrt(n * p * (1 - p))
    cov = np.einsum('bmi,bmj->bij', g, g) / blocks
    top = np.linalg.eigvalsh(cov)[:, -1]
    return float(np.quantile(top, NULL_QUANTILE, method='higher'))


def spectral_keep(obs: Observation, block_size: int):
    """Local block-gradient covariance filtering, followed by retained-data fit.

    The rejection budget is local to this encounter and therefore to each
    participating player. A block weight is shared by both players.
    """
    counts, n = block_counts(obs, block_size)
    blocks = len(counts)
    # Complementing either column or swapping them preserves eigenvalues.
    totals = counts.sum(axis=0)
    key = sorted(np.minimum(totals, blocks * n - totals).tolist())
    threshold = _conditional_threshold(blocks, n, *key)
    keep = np.ones(blocks, dtype=bool)
    trace = []
    budget = int(np.floor(MAX_REMOVAL * blocks))
    for iteration in range(budget + 1):
        active = np.flatnonzero(keep)
        g = standardized_scores(counts[keep], n)
        values, vectors = np.linalg.eigh(g.T @ g / len(g))
        top = float(values[-1])
        trace.append(top)
        if top <= threshold + 1e-12 or iteration == budget:
            break
        projections = (g @ vectors[:, -1]) ** 2
        keep[active[int(np.argmax(projections))]] = False
    return np.repeat(keep, block_size), {
        'threshold': threshold, 'eigenvalues': trace,
        'removed_blocks': int((~keep).sum()), 'block_keep': keep,
    }


def make_filter(cfg: Config, arm: str):
    # One synthetic session is one day. Match the desired per-session variance
    # to OnlineAbility's per-year convention without changing that class.
    return OnlineAbility({'hard': ALPHA}, sigma=cfg.drift * np.sqrt(365.25),
                         prior_var=cfg.prior_var,
                         robust_c=2.5 if arm == 'huber_2.5' else None)


def update(f: OnlineAbility, obs: Observation, keep: np.ndarray):
    t = date(2001, 1, 1) + timedelta(days=obs.session)
    for p in (obs.a, obs.b):
        f._ensure(p, t)
        f._diffuse(p, t)
    for side, server, returner in ((0, obs.a, obs.b), (1, obs.b, obs.a)):
        use = keep & (obs.serving == side)
        n = int(use.sum())
        if n:
            f._update_serve_obs(server, returner, int(obs.outcome[use].sum()), n, 'hard')


def align(values: np.ndarray):
    """Fix only the common serve/return gauge, independently each session."""
    return values - values.mean(axis=(-2, -1), keepdims=True)


def recovery_lags(estimate, truth, onset, tolerance=.2, sustained=3):
    error = np.abs(align(estimate) - align(truth))
    lags = []
    for player in (0, 1):
        good = (error[onset:, player] <= tolerance).all(axis=1)
        hits = [i for i in range(len(good) - sustained + 1)
                if good[i:i + sustained].all()]
        lags.append(hits[0] if hits else None)
    return lags


def run_world(cfg, generator, condition, seed):
    truth, observations, masks = world(cfg, generator, condition, seed)
    filters = {arm: make_filter(cfg, arm) for arm in ARMS}
    paths = {arm: np.empty_like(truth) for arm in ARMS}
    diagnostics = {arm: np.zeros(4) for arm in ARMS}
    block_example = []
    for obs, corrupted in zip(observations, masks):
        keeps = {'ordinary': np.ones(cfg.points, bool),
                 'huber_2.5': np.ones(cfg.points, bool), 'oracle': ~corrupted}
        for size in BLOCK_SIZES:
            keep, info = spectral_keep(obs, size)
            keeps[f'block_{size}'] = keep
            if seed == 0 and size == 16 and 0 in (obs.a, obs.b):
                block_example.append({'session': obs.session,
                                      'keep': info['block_keep'].tolist(),
                                      'corrupted_fraction': corrupted.reshape(-1, size).mean(axis=1).tolist(),
                                      'threshold': info['threshold'],
                                      'eigenvalues': info['eigenvalues']})
        for arm in ARMS:
            keep = keeps[arm]
            update(filters[arm], obs, keep)
            if obs.session >= cfg.onset:
                diagnostics[arm] += [np.sum(~keep & ~corrupted), np.sum(~corrupted),
                                     np.sum(~keep & corrupted), np.sum(corrupted)]
            f = filters[arm]
            for player in (obs.a, obs.b):
                paths[arm][obs.session, player] = [f.sm[player], f.rm[player]]
    metrics = {}
    for arm, path in paths.items():
        error = (align(path) - align(truth))[cfg.onset:]
        removed_clean, clean, removed_bad, bad = diagnostics[arm]
        metrics[arm] = {
            'focal_rmse': float(np.sqrt(np.mean(error[:, :2] ** 2))),
            'other_rmse': float(np.sqrt(np.mean(error[:, 2:] ** 2))),
            'all_rmse': float(np.sqrt(np.mean(error ** 2))),
            'per_player_rmse': np.sqrt(np.mean(error ** 2, axis=(0, 2))).tolist(),
            'removed_clean_fraction': float(removed_clean / clean),
            'removed_corrupted_fraction': float(removed_bad / bad) if bad else None,
            'recovery_lags': recovery_lags(path, truth, cfg.onset)
                             if condition.startswith('permanent') else None,
        }
    return {'generator': generator, 'condition': condition, 'seed': seed,
            'arms': metrics, 'block_example': block_example}, truth, paths


def interval(values):
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(8401)
    boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()),
            'lo': float(np.quantile(boot, .025)), 'hi': float(np.quantile(boot, .975))}


def summarize(cells):
    result = []
    for generator in GENERATORS:
        for condition in CONDITIONS:
            subset = [c for c in cells if c['generator'] == generator and c['condition'] == condition]
            for arm in ARMS:
                entry = {'generator': generator, 'condition': condition, 'arm': arm}
                for metric in ('focal_rmse', 'other_rmse', 'all_rmse',
                               'removed_clean_fraction', 'removed_corrupted_fraction'):
                    values = [c['arms'][arm][metric] for c in subset]
                    entry[metric] = interval(values) if values[0] is not None else None
                for group in ('focal', 'other'):
                    metric = f'{group}_rmse'
                    entry[f'{group}_gain'] = interval([c['arms']['ordinary'][metric] -
                                                       c['arms'][arm][metric] for c in subset])
                entry['per_player_rmse'] = [interval([c['arms'][arm]['per_player_rmse'][p]
                                                     for c in subset]) for p in range(8)]
                if condition.startswith('permanent'):
                    lags_by_seed = [c['arms'][arm]['recovery_lags'] for c in subset]
                    rates = [sum(v is not None for v in lags) / 2 for lags in lags_by_seed]
                    observed = [v for lags in lags_by_seed for v in lags if v is not None]
                    # Resample whole worlds, preserving both focal players.
                    rng = np.random.default_rng(8401)
                    boot = []
                    for indices in rng.integers(len(subset), size=(2000, len(subset))):
                        samples = [v for i in indices for v in lags_by_seed[i] if v is not None]
                        if samples:
                            boot.append(float(np.mean(samples)))
                    entry['recovery'] = {'fraction': interval(rates),
                                         'recovered': len(observed), 'total': 2 * len(subset),
                                         'lag_among_recovered': {
                                             'mean': float(np.mean(observed)),
                                             'lo': float(np.quantile(boot, .025)),
                                             'hi': float(np.quantile(boot, .975))} if observed else None}
                result.append(entry)
    return result


def fmt(value):
    return '—' if value is None else f"{value['mean']:.4f} [{value['lo']:.4f}, {value['hi']:.4f}]"


def report(result, out):
    lines = ['# Ordered-point robustness pilot', '',
             f"Generated {result['utc']}; {result['config']['seeds']} paired seeds per cell.", '',
             'Exploratory mechanism experiment; no real data or 2025 test used. '
             'Settings were fixed before execution; no block size is selected as a winner. '
             'See [design](../../docs/POINT_ROBUST_PILOT.md).', '',
             'RMSE is in ability logits, after fixing only the shared serve/return offset. '
             'Evaluation starts at session 24. Focal = players 0 and 1; others = remaining players. '
             'Positive paired gain means lower RMSE than ordinary. Intervals are 95% paired '
             'world-bootstrap intervals (Monte Carlo uncertainty, not real-data confidence).', '',
             '## Ability recovery', '',
             '| Truth | Condition | Arm | Focal RMSE [CI] | Focal gain [CI] | Other-player RMSE [CI] | Other-player gain [CI] |',
             '|---|---|---|---|---|---|---|']
    for r in result['summary']:
        lines.append(f"| {r['generator']} | {r['condition']} | {r['arm']} | "
                     f"{fmt(r['focal_rmse'])} | {fmt(r['focal_gain'])} | "
                     f"{fmt(r['other_rmse'])} | {fmt(r['other_gain'])} |")
    lines += ['', '## Removed point fractions', '',
              'These describe explicit data rejection. Huber clips scores and does not remove points.', '',
              '| Truth | Condition | Arm | Clean points removed [CI] | Disrupted points removed [CI] |',
              '|---|---|---|---|---|']
    for r in result['summary']:
        if r['condition'] in ('clean', 'burst') and r['arm'].startswith('block'):
            lines.append(f"| {r['generator']} | {r['condition']} | {r['arm']} | "
                         f"{fmt(r['removed_clean_fraction'])} | {fmt(r['removed_corrupted_fraction'])} |")
    lines += ['', '## Sustained-change adaptation', '',
              'Recovery = both ability errors ≤0.2 logits for three consecutive sessions. '
              'Lag counts from the change to the start of that run. Non-recovery is censored; '
              'mean lag is conditional on recovery and must be read with recovery fraction.', '',
              '| Truth | Change | Arm | Recovered / total | Recovery fraction [CI] | Lag among recovered [CI] |',
              '|---|---|---|---|---|---|']
    for r in result['summary']:
        if 'recovery' in r:
            d = r['recovery']
            lines.append(f"| {r['generator']} | {r['condition']} | {r['arm']} | "
                         f"{d['recovered']} / {d['total']} | {fmt(d['fraction'])} | "
                         f"{fmt(d['lag_among_recovered'])} |")
    lines += ['', '## Preselected illustration', '',
              'Random-walk seed 0, player 0, block size 16; chosen before execution. '
              'All three block sizes remain in the tables above.', '',
              '![Ability trajectories](trajectories.png)', '',
              '![Block rejection and planted disruption](block_weights.png)', '',
              '## Limits', '',
              'This tests within-encounter heterogeneity. A uniformly impaired encounter can '
              'look like a genuine level change. Short or weak bursts may be undetectable. '
              'The spectral prefilter is SEVER-inspired, not published SEVER/MMW, and carries '
              'no inherited arbitrary-contamination guarantee. Its clean calibration assumes '
              'independent, constant-probability points within each service stream. '
              'Natural dependence could trigger rejection on real data.', '',
              'All arms retain the existing approximate online ability backbone. The pilot '
              'uses balanced service sequences and fixed encounter lengths, with no scoring, '
              'bracket selection, surface changes or missingness. It does not validate the '
              'full tennis pipeline. The oracle is a known-mask reference, not a guaranteed '
              'finite-sample error bound. Individual-player results and all seed-level '
              'metrics are in results.json; raw fitted trajectories are in trajectories.npz.', '']
    (out / 'report.md').write_text('\n'.join(lines))


def plot(examples, cells, cfg, out):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tennis_point_robust_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'ordinary': '#cc6633', 'huber_2.5': '#a585bd',
              'block_16': '#007f8b', 'oracle': '#4b9e53'}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    for row, condition in enumerate(('clean', 'burst', 'permanent_drop')):
        truth, paths = examples[condition]
        for side, skill in enumerate(('Serve', 'Return')):
            ax = axes[row, side]
            ax.plot(align(truth)[:, 0, side], color='#242424', linewidth=2.2, label='True skill')
            for arm, color in colors.items():
                ax.plot(align(paths[arm])[:, 0, side], label=arm, color=color,
                        alpha=.9, linestyle='--' if arm == 'oracle' else '-')
            ax.axvline(cfg.onset, color='#777777', linewidth=.8, linestyle=':')
            if condition == 'burst':
                ax.axvspan(cfg.onset, cfg.burst_end - 1, color='#e6bc71', alpha=.2)
            ax.set_title(f"{condition.replace('_', ' ').title()} · {skill}")
            ax.set_ylabel('Ability (logit)')
            ax.grid(alpha=.15)
            if row == 2:
                ax.set_xlabel('Session (estimate after observing points)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
    fig.suptitle('Underlying skill recovery · preselected random-walk seed 0, player 0')
    fig.savefig(out / 'trajectories.png', dpi=170)
    plt.close(fig)
    example = next(c for c in cells if c['generator'] == 'rw' and
                   c['condition'] == 'burst' and c['seed'] == 0)['block_example']
    bad = np.array([x['corrupted_fraction'] for x in example]).T
    removed = 1 - np.array([x['keep'] for x in example], dtype=int).T
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
    for ax, data, title in zip(axes, (bad, removed),
                               ('Planted disruption: fraction of points per block',
                                'Spectral filter: rejected blocks (1 = rejected)')):
        im = ax.imshow(data, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='magma_r')
        ax.set_title(title)
        ax.set_ylabel('16-point block')
        fig.colorbar(im, ax=ax, shrink=.8)
    axes[-1].set_xlabel('Session')
    fig.suptitle('Player 0 encounters · truth labels are used only for evaluation')
    fig.savefig(out / 'block_weights.png', dpi=170)
    plt.close(fig)


def run(cfg: Config, out: Path):
    if cfg.seeds < 2:
        raise ValueError('at least two independent seeds are required')
    if out.exists():
        raise FileExistsError(f'{out} already exists; use a new --out directory')
    ensure(out)
    sources = ['src/experiments/point_robustness.py', 'src/models/ability.py',
               'docs/POINT_ROBUST_PILOT.md']
    metadata = {'utc': utcnow(), 'config': asdict(cfg),
                'method': {'block_sizes': BLOCK_SIZES, 'max_removal': MAX_REMOVAL,
                           'null_draws': NULL_DRAWS, 'null_quantile': NULL_QUANTILE},
                'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                  for p in sources}}
    write_json(out / 'config.json', metadata)
    cells, trajectories, examples = [], {}, {}
    for generator in GENERATORS:
        for condition in CONDITIONS:
            for seed in range(cfg.seeds):
                ids = {arm: trials.log(stage='point-robust-pilot', kind=arm,
                                      params={**metadata['config'], **metadata['method'],
                                              'generator': generator, 'condition': condition,
                                              'seed': seed, 'output': str(out),
                                              'source_sha256': metadata['source_sha256']},
                                      note='Started fixed exploratory configuration; no real data.')
                       for arm in ARMS}
                cell, truth, paths = run_world(cfg, generator, condition, seed)
                cell['trial_ids'] = ids
                cells.append(cell)
                stem = f'{generator}_{condition}_{seed}'
                write_json(out / 'cells' / f'{stem}.json', cell)
                trajectories[f'{stem}_truth'] = truth
                for arm, path in paths.items():
                    trajectories[f'{stem}_{arm}'] = path
                if generator == 'rw' and seed == 0:
                    examples[condition] = (truth, paths)
                print(f'{generator}/{condition} seed {seed + 1}/{cfg.seeds}', flush=True)
    result = {**metadata, 'summary': summarize(cells), 'cells': cells}
    write_json(out / 'results.json', result)
    np.savez_compressed(out / 'trajectories.npz', **trajectories)
    plot(examples, cells, cfg, out)
    report(result, out)
    print(f'Report: {out / "report.md"}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--out', type=Path, default=RESULTS / 'point_robust_pilot')
    args = parser.parse_args()
    run(Config(seeds=args.seeds), args.out)
