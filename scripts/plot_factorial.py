"""Plot aggregate synthetic policy differences from the factorial summary CSV."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(source, output):
    with Path(source).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows = [r for r in rows if r['analysis'] == 'E2_A_minus_G']
    conditions = [(added, quantity, prevalence)
                  for added in ('no_information', 'duplicated', 'complementary')
                  for quantity in (False, True) for prevalence in (.5, .3)]
    matrices = []
    for arm in ('text', 'quantity'):
        values = np.full((3, 12), np.nan)
        for r in rows:
            if r['arm'] != arm:
                continue
            if r['status'] != 'complete' or int(r['n_seeds']) != 30:
                raise ValueError('Plot requires complete 30-seed cells')
            i = (.6, .7, .8).index(float(r['base_strength']))
            key = (r['added_type'], r['quantity'].lower() == 'true', float(r['positive_fraction']))
            j = conditions.index(key)
            if np.isfinite(values[i, j]):
                raise ValueError('Duplicate synthetic condition')
            values[i, j] = float(r['delta_auroc_mean_across_seeds'])
        if not np.isfinite(values).all():
            raise ValueError('Missing synthetic condition')
        matrices.append(values)
    extent = max(.01, max(float(np.max(np.abs(m))) for m in matrices))
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.7), layout='constrained')
    labels = [('Noise' if a == 'no_information' else 'Copy' if a == 'duplicated' else 'Complement')
              + '\nQ' + ('on' if q else 'off') + '/' + str(int(p*100)) + '%'
              for a, q, p in conditions]
    for ax, arm, values in zip(axes, ('TF-IDF text', 'Token quantity'), matrices):
        im = ax.imshow(values, cmap='RdBu_r', vmin=-extent, vmax=extent, aspect='auto')
        ax.set_xticks(range(12), labels, fontsize=8)
        ax.set_yticks(range(3), ['0.60', '0.70', '0.80'])
        ax.set_ylabel('Base sensitivity / specificity')
        ax.set_title(arm)
        for i in range(3):
            for j in range(12):
                ax.text(j, i, f'{values[i,j]:+.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(values[i,j]) > .55*extent else 'black')
    fig.colorbar(im, ax=axes, label='Mean AUROC difference: absolute minus gain selection', shrink=.85)
    fig.suptitle('Synthetic factorial: 36 conditions, 30 independent dataset seeds per cell\nQ = response-quantity signal; percent = Group 1 prevalence', fontsize=11)
    output = Path(output);output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight');plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)
