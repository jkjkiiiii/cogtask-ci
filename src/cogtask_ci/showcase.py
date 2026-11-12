"""Four prespecified synthetic sentence scenarios and a reproducible overview."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from .evaluation import Config, evaluate

SCENARIOS = ('complementary', 'redundant', 'content_quantity', 'null')
TASKS = ('task_a', 'task_b', 'task_c')
SIGNAL = (
    'The window is closed and the room looks quiet today.',
    'The window is open and the room looks bright today.',
)
NEUTRAL = 'A person follows the path and passes a wooden gate.'


def scenario_rows(scenario, subjects=160, seed=20261103):
    if scenario not in SCENARIOS:
        raise ValueError('Unknown synthetic scenario')
    if type(subjects) is not int or subjects < 24 or subjects % 2:
        raise ValueError('Synthetic subjects must be even and >=24')
    rng = np.random.default_rng(seed)
    result = []
    for i in range(subjects):
        label = i % 2
        shared = int(rng.random() < (.8 if label else .2))
        for task in TASKS:
            if scenario in ('complementary', 'content_quantity'):
                signal = int(rng.random() < (.8 if label else .2))
            elif scenario == 'redundant':
                signal = shared
            else:
                signal = int(rng.random() < .5)
            segments = [NEUTRAL.split(), SIGNAL[signal].split()]
            if scenario == 'content_quantity':
                # Length and lexical markers are independent conditional on the
                # label. Both are informative, with different fixed strengths.
                copies = 4 if rng.random() < (.65 if label else .35) else 1
                segments = segments * copies
            result.append(dict(subject_id=f'S{i:04d}', task=task, label=label, segments=segments))
    return result


def plot_showcase(rows, output, subjects, config, contrasts=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(12, 8.8), facecolor='#ffffff')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    titles = ('A  Complementary signals', 'B  Duplicated task information',
              'C  Beyond response quantity', 'D  No designed label signal')
    subtitles = ('Independent noisy lexical markers across tasks',
                 'The same response is reused across all three tasks',
                 'Lexical information supplements an informative length cue',
                 'Lexical choices are sampled independently of labels')
    colors = {'text': '#2463a7', 'quantity': '#da8534'}
    for ax, scenario, title, subtitle in zip(axes.flat, SCENARIOS, titles, subtitles):
        for arm in ('text', 'quantity'):
            selected = sorted((r for r in rows if r['scenario'] == scenario and r['arm'] == arm), key=lambda r: int(r['k']))
            k = [int(r['k']) for r in selected]
            y = [float(r['auroc']) for r in selected]
            ax.plot(k, y, color=colors[arm], marker='o' if arm == 'text' else 's',
                    linestyle='-' if arm == 'text' else '--', linewidth=2.1, markersize=6,
                    label='Text model' if arm == 'text' else 'Token-count model')
            ax.fill_between(k, [float(r['ci_low']) for r in selected], [float(r['ci_high']) for r in selected],
                            color=colors[arm], alpha=.12 if arm == 'text' else .07, linewidth=0)
        ax.axhline(.5, color='#89939e', linestyle=':', linewidth=1.2)
        ax.set(xticks=[1, 2, 3], xlabel='Number of tasks', ylabel='Outer-OOF AUROC', ylim=(0, 1), xlim=(.85, 3.15))
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=29, color='#172f46')
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9, color='#566779')
        ax.grid(axis='y', color='#e9edf2', linewidth=.7)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.legend(loc='lower right', fontsize=8, frameon=False)
        if scenario == 'content_quantity' and contrasts:
            pair = next(r for r in contrasts if r['scenario'] == scenario and int(r['k']) == 3)
            ax.text(.035, .30,
                    f"Text − count at 3 tasks: {float(pair['auroc_difference']):+.3f}\n"
                    f"Paired 95% CI [{float(pair['ci_low']):+.3f}, {float(pair['ci_high']):+.3f}]",
                    transform=ax.transAxes, fontsize=9, color='#172f46', linespacing=1.5)
    fig.suptitle('CogTask-CI  |  Synthetic examples', x=.07, y=.985, ha='left', fontsize=21, fontweight='bold', color='#172f46')
    fig.text(.07, .94, f'{subjects} artificial participants per scenario  ·  {config.outer_folds}-fold outer / {config.inner_folds}-fold inner CV  ·  {config.bootstrap} participant bootstrap draws', fontsize=10, color='#566779')
    fig.text(.07, .025, 'Bands: conditional pointwise 95% intervals. One fixed simulation seed; no outcome-based seed selection.\nArtificial Group 0/1 labels and templated sentences — not study data or clinical performance.', fontsize=9, color='#566779', linespacing=1.6)
    fig.subplots_adjust(left=.07, right=.975, top=.84, bottom=.14, hspace=.62, wspace=.22)
    for extension in ('png', 'svg'):
        fig.savefig(Path(output)/('showcase.' + extension), dpi=170, facecolor='white')
    fig.clear()


def build_showcase(output, subjects=160, seed=20261103, bootstrap=500, assets=None, replace_assets=False):
    output = Path(output)
    if output.exists():
        raise ValueError('Showcase output must not exist')
    if assets:
        assets = Path(assets)
        names = ['showcase.png', 'showcase.svg', 'showcase_summary.csv', 'showcase_contrasts.csv', 'showcase_config.json']
        if not replace_assets and any((assets/name).exists() for name in names):
            raise ValueError('Showcase assets already exist; explicitly pass --replace-assets to regenerate')
    config = Config(outer_folds=3, inner_folds=2, bootstrap=bootstrap, seed=42)
    config.validate()
    # Validate generation arguments before creating a run directory.
    scenario_rows('null', subjects, seed)
    output.mkdir(parents=True)
    summary = []
    paired = []
    for scenario in SCENARIOS:
        data = output/(scenario + '.jsonl')
        data.write_text('\n'.join(json.dumps(r) for r in scenario_rows(scenario, subjects, seed)) + '\n', encoding='utf-8')
        print('Evaluating synthetic scenario: ' + scenario, flush=True)
        metrics = evaluate(data, output/scenario, config)
        summary.extend(dict(scenario=scenario, **row) for row in metrics)
        with (output/scenario/'contrasts.csv').open(encoding='utf-8-sig', newline='') as f:
            paired.extend(dict(scenario=scenario, **row) for row in csv.DictReader(f))
    with (output/'showcase_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output/'showcase_contrasts.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    metadata = dict(synthetic_only=True, subjects_per_scenario=subjects, generator_seed=seed,
                    scenarios={
                        'complementary': {'lexical_probability_by_group': {'0': .2, '1': .8}, 'tasks': 'conditionally independent markers', 'length': 'constant'},
                        'redundant': {'lexical_probability_by_group': {'0': .2, '1': .8}, 'tasks': 'identical response copies', 'length': 'constant'},
                        'content_quantity': {'lexical_probability_by_group': {'0': .2, '1': .8}, 'long_response_probability_by_group': {'0': .35, '1': .65}, 'copies': [1, 4], 'independence': 'lexical and length draws independent conditional on label'},
                        'null': {'lexical_probability_by_group': {'0': .5, '1': .5}, 'length': 'constant'}},
                    scenario_order=list(SCENARIOS), showcase_design_version=2,
                    task_order=list(TASKS), evaluation=asdict(config),
                    selection_note='One prespecified seed/configuration; no selection based on resulting scores')
    (output/'showcase_config.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    plot_showcase(summary, output, subjects, config, paired)
    if assets:
        assets.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(output/name, assets/name)
    return output/'showcase.png'
