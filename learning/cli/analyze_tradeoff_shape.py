"""Exploratory equal-weight shape analysis of two saved terrain conditions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fit(name, x, y):
    if name in ('linear', 'reciprocal', 'fixed_speed_square_root'):
        z = 1 / x if name == 'reciprocal' else x
        target = (2.6-y)**2 if name == 'fixed_speed_square_root' else y
        params = np.linalg.lstsq(np.column_stack((np.ones(len(x)), z)), target, rcond=None)[0]
        if name == 'fixed_speed_square_root':
            predict = lambda q: 2.6 - np.sqrt(params[0] + params[1] * q)
        else:
            predict = lambda q: params[0] + params[1] * (1/q if name == 'reciprocal' else q)
    else:
        function = (lambda q, a, b: a*q**(-b)) if name == 'power' else (lambda q, a, b: a*np.exp(-b*q))
        params, _ = curve_fit(function, x, y, p0=(.05, .5) if name == 'power' else (.2, 5),
                              bounds=(0, np.inf), maxfev=10000)
        predict = lambda q: function(q, *params)
    return params, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    names = ('oblique_2cm', 'oblique_4cm')
    cases = json.loads((args.evaluation/'index.json').read_text())
    selected = [next(c for c in cases if c['terrain'] == name and c['user_speed_m_s'] == 2.6) for name in names]
    traces = [np.load(Path(c['run'])/'trajectory.npz')['lower_trace'].reshape(-1, 6, 11) for c in selected]
    mask = np.logical_and.reduce([t[:, :, 6].astype(bool).all(1) for t in traces])
    points = np.array([np.stack((np.sqrt(np.mean(t[mask, :5, 3]**2, axis=0)),
                                np.sqrt(np.mean((t[mask, :5, 2]-2.6)**2, axis=0))), axis=-1) for t in traces])
    weights = np.array([.5, .5])
    mean = np.einsum('i,ijk->jk', weights, points)
    pooled = np.sqrt(np.einsum('i,ijk->jk', weights, points**2))
    x, y = mean.T
    models = ['linear', 'reciprocal', 'power', 'exponential', 'fixed_speed_square_root']
    results = {}
    for name in models:
        params, predict = fit(name, x, y)
        held_out = []
        for i in range(5):
            keep = np.arange(5) != i
            _, loo_predict = fit(name, x[keep], y[keep])
            held_out.append(float(loo_predict(x[i])))
        residual = y-predict(x)
        results[name] = dict(parameters=params.tolist(), rmse=float(np.sqrt(np.mean(residual**2))),
            r2=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
            leave_one_alpha_out_predictions=held_out,
            leave_one_alpha_out_rmse=float(np.sqrt(np.mean((y-np.array(held_out))**2))))
    sensitivity = {}
    for label, keep in [('all', np.ones(5, bool)), ('omit_alpha0', np.arange(5)!=0),
                        ('omit_alpha1', np.arange(5)!=4)]:
        params, predict = fit('reciprocal', x[keep], y[keep])
        sensitivity[label] = dict(parameters=params.tolist(), predictions_all_five=predict(x).tolist())
    separate = {}
    for name, coords in zip(names, points):
        xx, yy = coords.T
        params, predict = fit('reciprocal', xx, yy)
        separate[name] = dict(parameters=params.tolist(),
            rmse=float(np.sqrt(np.mean((yy-predict(xx))**2))),
            r2=float(1-np.sum((yy-predict(xx))**2)/np.sum((yy-yy.mean())**2)))
    weight_sensitivity = {}
    for w in (.25, .5, .75):
        coords = w*points[0]+(1-w)*points[1]
        params, _ = fit('reciprocal', *coords.T)
        weight_sensitivity[str(w)] = params.tolist()
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8A6500']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    for i, terrain in enumerate(names):
        axes[0].plot(*points[i].T, linestyle=['--', ':'][i], color='.6', label=terrain)
    axes[0].plot(x, y, color='.25', label='Equal-weight mean')
    grid = np.linspace(x.min(), x.max(), 200)
    for name, style in [('reciprocal', '-'), ('power', '--'), ('exponential', ':')]:
        _, predict = fit(name, x, y)
        axes[1].plot(grid, predict(grid), linestyle=style, label=name)
    _, predict = fit('fixed_speed_square_root', x, y)
    axes[1].plot(grid, predict(grid), '-.', label='Fixed-speed sqrt hypothesis')
    for label, values in sensitivity.items():
        p = values['parameters']
        axes[2].plot(grid, p[0]+p[1]/grid, label=label)
    for ax in axes:
        for alpha, xx, yy, color, marker in zip([0,.25,.5,.75,1],x,y,colors,['o','s','^','D','v']):
            ax.scatter(xx, yy, color=color, marker=marker, s=40, zorder=4)
        ax.set(xlabel='Path RMSE (m)', ylabel='Speed RMSE (m/s)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_title('Two terrains and their mean (five alphas)')
    axes[1].set_title('Descriptive two-parameter curves')
    axes[2].set_title('Endpoint omission: sensitivity only')
    fig.suptitle('Original H75 | request 2.6 m/s | identical shared time mask across both terrains\n'
                 'All five alpha points retained in main analysis; no uncertainty or physical-law claim.', fontsize=11)
    fig.savefig(args.evaluation/'tradeoff_shape_analysis.png', dpi=160)
    plt.close(fig)
    record = dict(role='Post-hoc descriptive DEV analysis; no new simulations', source_cases=selected,
        weights=weights.tolist(), weighting_reason='Equal scenario importance; no independent replicates for inverse-variance weights',
        sample_count=int(mask.sum()), window_end_s=float((np.flatnonzero(mask)[-1]+1)*.005),
        alphas=[0,.25,.5,.75,1], terrain_points=points.tolist(), mean_rmse_coordinates=mean.tolist(),
        pooled_rmse_coordinates=pooled.tolist(), fits=results, reciprocal_sensitivity=sensitivity,
        separate_terrain_reciprocal=separate, first_terrain_weight_sensitivity=weight_sensitivity,
        exclusion='None in main analysis. Endpoint omissions are sensitivity, not outlier labels.',
        model_note='Sqrt fits squared speed proxy, other fits minimize speed-error squared residual; LOO scored in speed-error units. Five deterministic alpha points are not random replicates.')
    (args.evaluation/'tradeoff_shape_analysis.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(mean=mean.tolist(),fits=results,sensitivity=sensitivity),indent=2))


if __name__ == '__main__':
    main()
