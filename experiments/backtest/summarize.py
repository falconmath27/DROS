"""Regenerate P5 tables and report fragments from the completed frozen grid."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.bootstrap import load_fold_results, analyze_comparison_families

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calendar_slice(dates):
    """Descriptive calendar slices, not an inferred calm/stress classification."""
    years = pd.to_datetime(dates).dt.year
    return np.where(years == 2020, '2020', np.where(years == 2022, '2022', 'other years'))


def render_tables(out):
    """Render compact descriptive differences from the generated CSV tables."""
    for source, name, keys in (
        ('market_cost_tables.csv', 'costs.tex', ['market','objective','cost']),
        ('regime_tables.csv', 'regimes.tex', ['market','objective','regime']),
    ):
        frame = pd.read_csv(out/source).query("metric == 'net_return'")
        if name == 'regimes.tex':
            frame = frame.query('cost == 1')
        wide = frame.pivot(index=keys, columns='method', values='mean')
        wide['GRU minus MLP'] = wide['OSPG-GRU']-wide['OSPG-MLP']
        wide[['OSPG-GRU','OSPG-MLP','GRU minus MLP']].reset_index().to_latex(
            out/name, index=False, escape=True, float_format='%.4f')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT/'results/p5/full_grid')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/p5/summary')
    parser.add_argument('--n-boot', type=int, default=2000)
    args = parser.parse_args(argv)
    if args.n_boot < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    run, out = args.run_root, args.output_root
    coverage = json.loads((run/'coverage.json').read_text())
    if not coverage['passed'] or coverage['scope']['smoke']:
        raise ValueError('A complete, non-smoke frozen grid is required')
    config_path = ROOT/'experiments/backtest/config.json'
    config = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'n_boot': args.n_boot, 'bootstrap_seed': 0, 'inputs': {}}
    def track(path):
        manifest['inputs'][str(path.resolve())] = digest(path)
    track(config_path)
    track(Path(__file__))
    track(ROOT/'analysis/bootstrap.py')
    track(run/'coverage.json')
    means, regimes, contrasts, persistence = [], [], [], []
    for market in config['markets']:
        price_path = ROOT/config['markets'][market]['inputs']['clean_prices']['path']
        track(price_path)
        if digest(price_path) != config['markets'][market]['inputs']['clean_prices']['sha256']:
            raise ValueError('Frozen price hash mismatch')
        prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        returns = np.log(prices).diff()
        for objective in ('linear', 'exponential'):
            for cost in (0, 1, 2):
                scenario = run/market/objective/f'cost_{cost}x'
                resolved = json.loads((scenario/'resolved_config.json').read_text())
                if resolved['source_config_sha256'] != digest(config_path):
                    raise ValueError('Config differs from frozen run')
                track(scenario/'resolved_config.json')
                frames = {'objective': [], 'net_return': []}
                for fold in range(config['markets'][market]['dataset']['folds']):
                    for seed in config['seeds']:
                        path = scenario/f'fold_{fold}_seed_{seed}.npz'
                        track(path)
                        status_path = path.with_suffix('.status.json')
                        track(status_path)
                        status = json.loads(status_path.read_text())
                        if not (status['deadline_purge']['train_deadline_before_validation'] and
                                status['deadline_purge']['validation_deadline_before_test']):
                            raise ValueError('Deadline purge failed')
                        if any(status['methods'][m]['status'] != 'completed' or
                               not status['methods'][m]['prefix_causal'] for m in config['methods']):
                            raise ValueError('Incomplete or noncausal method')
                        for metric in frames:
                            frames[metric].append(load_fold_results(path, market=market,
                                methods=config['methods'], metric=metric))
                for metric, pieces in frames.items():
                    data = pd.concat(pieces, ignore_index=True)
                    label = dict(market=market, objective=objective, cost=cost, metric=metric)
                    averaged = data.groupby(['fold','window_id','entry_date','asset_id','method'], as_index=False).score.mean()
                    for method, group in averaged.groupby('method'):
                        means.append(dict(**label, method=method, mean=group.score.mean(), n_windows=len(group)))
                        group = group.assign(regime=calendar_slice(group.entry_date))
                        for regime, part in group.groupby('regime'):
                            regimes.append(dict(**label, method=method, regime=regime,
                                mean=part.score.mean(), n_windows=len(part), n_dates=part.entry_date.nunique()))
                    print(f'{market} {objective} cost={cost} {metric}: bootstrap', flush=True)
                    families = analyze_comparison_families(data, config, n_boot=args.n_boot)
                    for family, rows in families.items():
                        for row in rows:
                            contrasts.append(dict(**label, **row))
                    if cost == 1 and metric == 'net_return':
                        pivot = averaged.pivot(index=['fold','window_id','entry_date','asset_id'], columns='method', values='score').reset_index()
                        pivot['advantage'] = pivot['OSPG-GRU']-pivot['OSPG-MLP']
                        folds = json.loads((scenario/'folds.json').read_text())
                        track(scenario/'folds.json')
                        for (fold, asset), part in pivot.groupby(['fold','asset_id']):
                            boundary = next(f for f in folds if f['number'] == fold)['validation_start']
                            x = returns.loc[returns.index < pd.Timestamp(boundary), asset].dropna().abs()
                            rho = x.autocorr(1) if len(x) >= 30 and x.std() > 0 else np.nan
                            persistence.append(dict(market=market, objective=objective, fold=fold,
                                asset=asset, persistence=rho, advantage=part.advantage.mean(), n_windows=len(part)))
    pd.DataFrame(means).to_csv(out/'market_cost_tables.csv', index=False)
    pd.DataFrame(regimes).to_csv(out/'regime_tables.csv', index=False)
    (out/'paired_statistics.json').write_text(json.dumps(contrasts, indent=2), encoding='utf-8')
    stats = pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in contrasts])
    stats.to_csv(out/'paired_statistics.csv', index=False)
    diagnostics = pd.DataFrame(persistence)
    diagnostics.to_csv(out/'persistence.csv', index=False)
    fig, axes = plt.subplots(1,2,figsize=(10,4), sharey=True)
    for ax, objective in zip(axes, ('linear','exponential')):
        for market, group in diagnostics[diagnostics.objective == objective].groupby('market'):
            ax.scatter(group.persistence, group.advantage, s=9, alpha=.35, label=market)
        ax.axhline(0,color='black',linewidth=.7)
        ax.set(title=objective, xlabel='Training |log return| lag-1 autocorrelation')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('GRU minus MLP mean economic net return')
    fig.tight_layout()
    fig.savefig(out/'persistence.png', dpi=180)
    plt.close(fig)
    primary = stats[(stats.cost == 1)&(stats.comparison_family == 'primary')]
    table = primary[['market','objective','metric','estimate','ci_lower','ci_upper','p_value_holm']]
    table.to_latex(out/'primary.tex', index=False, escape=True, float_format='%.6f')
    pd.DataFrame(means).query('cost == 1').to_latex(out/'markets.tex', index=False, escape=True,float_format='%.4f',longtable=True)
    summary = []
    for row in primary.itertuples():
        summary.append(f'{row.market}, {row.objective}, {row.metric}: GRU minus MLP {row.estimate:.5f}, 95% CI [{row.ci_lower:.5f}, {row.ci_upper:.5f}], adjusted p={row.p_value_holm:.4f}.')
    (out/'findings.txt').write_text('\n'.join(summary),encoding='utf-8')
    narrative = []
    for row in primary.itertuples():
        direction = 'higher' if row.estimate > 0 else 'lower'
        evidence = 'excludes zero' if row.ci_lower > 0 or row.ci_upper < 0 else 'includes zero'
        metric_label = 'economic net return' if row.metric == 'net_return' else 'objective score'
        narrative.append(f'{row.market} under {row.objective} training: GRU has {direction} mean {metric_label} than MLP; the interval {evidence}.')
    (out/'interpretation.tex').write_text('\n\n'.join(narrative), encoding='utf-8')
    render_tables(out)
    manifest['outputs'] = {p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return primary


if __name__ == '__main__':
    main()
