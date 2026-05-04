"""
Usage:

sbatch runtime/clusters/pegasus/shell/run.sh --no-gpu "\
python -m scripts.evals.quant_boot_diff_sci \
    --dir-name batch_3_16_rand_5 \
    --source-lang-group joshi5 \
    --group-by lang \
    --bootstrap \
    --n-resamples 1000 \
    --worker 64 \
"& \
sbatch runtime/clusters/pegasus/shell/run.sh --no-gpu "\
python -m scripts.evals.quant_boot_diff_sci \
    --dir-name batch_4_16_rand_5 \
    --source-lang-group joshi5 \
    --group-by lang \
    --bootstrap \
    --n-resamples 1000 \
    --worker 64 \
"

squeue -u bmikaberidze -l  
"""

import csv
import os
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import bootstrap as scipy_bootstrap

from micm_nlp.models.model import MODEL
from scripts.evals.utils import extract_grouped_hs_embeddings

# ── Config ──────────────────────────────────────────────────────────────────────
GROUP_BY = 'script'  # 'lang' | 'script'
DIR_NAME = 'batch_1_5_rand_1'
SOURCE_LANG_GROUP = 'joshi5'
# 
N_LAYERS = 25
SIL_N_SAMPLES = 0  # None = full, 0 = skip, int = subsample
SIL_DIST_METRIC = 'manhattan'
N_RESAMPLES = 1000
# 
IN_PATH = f'{MODEL.stor_path}/xlmr/FacebookAI|xlm-roberta-large/prompt_hidden_states'
OUT_PATH = 'artefacts/evals/hs_quant'
# 
METHODS = ('xpe', 'spt')
METRICS = ('sil', 'db', 'ch')
BOOT_METRICS = ('db', 'ch')  # skip sil for bootstrap
META = ('n_samples', 'n_clusters')

def get_csv_columns(use_bootstrap=False):
    cols = ['file_name'] + [
        f'{method}_{field}'
        for field in [*METRICS, *META]
        for method in METHODS
    ] + [f'diff_{m}' for m in METRICS]

    if use_bootstrap:
        for metric in BOOT_METRICS:
            cols.append(f'diff_{metric}_ci_l')
            cols.append(f'diff_{metric}_ci_u')
            cols.append(f'diff_{metric}_se')
            cols.append(f'diff_{metric}_p')

    return cols


# ── Core ────────────────────────────────────────────────────────────────────────
def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict:

    unique = np.unique(labels)
    numeric = np.array([{l: i for i, l in enumerate(unique)}[l] for l in labels])

    sil = 0.0
    if SIL_N_SAMPLES is None:
        sil = silhouette_score(embeddings, numeric, metric=SIL_DIST_METRIC)
    elif SIL_N_SAMPLES:
        idx = resample(np.arange(len(embeddings)), n_samples=SIL_N_SAMPLES, random_state=42)
        sil = silhouette_score(embeddings[idx], numeric[idx], metric=SIL_DIST_METRIC)

    return {
        'sil': sil,
        'db': davies_bouldin_score(embeddings, numeric),
        'ch': calinski_harabasz_score(embeddings, numeric),
        'n_samples': len(embeddings),
        'n_clusters': len(unique),
    }


def bootstrap_difference(xpe_emb, xpe_labels, spt_emb, spt_labels, n_resamples):
    """Bootstrap the difference (XPE - SPT) for DB and CH. Returns CI, SE, p-value."""
    xpe_unique = np.unique(xpe_labels)
    xpe_numeric = np.array([{l: i for i, l in enumerate(xpe_unique)}[l] for l in xpe_labels])
    spt_unique = np.unique(spt_labels)
    spt_numeric = np.array([{l: i for i, l in enumerate(spt_unique)}[l] for l in spt_labels])

    result = {}

    for metric_name, metric_fn in [('db', davies_bouldin_score),
                                    ('ch', calinski_harabasz_score)]:

        def statistic(xpe_idx, spt_idx,
                      _xpe_emb=xpe_emb, _xpe_num=xpe_numeric,
                      _spt_emb=spt_emb, _spt_num=spt_numeric,
                      _fn=metric_fn):
            xpe_idx = xpe_idx.astype(int)
            spt_idx = spt_idx.astype(int)
            return _fn(_xpe_emb[xpe_idx], _xpe_num[xpe_idx]) - _fn(_spt_emb[spt_idx], _spt_num[spt_idx])

        boot_result = scipy_bootstrap(
            (np.arange(len(xpe_emb)), np.arange(len(spt_emb))),
            statistic,
            n_resamples=n_resamples,
            confidence_level=0.95,
            method='percentile',
            random_state=42,
            paired=False,
        )

        # p-value: fraction of bootstrap diffs on wrong side of zero
        diffs = boot_result.bootstrap_distribution
        observed_diff = metric_fn(xpe_emb, xpe_numeric) - metric_fn(spt_emb, spt_numeric)
        if observed_diff > 0:
            p_value = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)
        else:
            p_value = (np.sum(diffs >= 0) + 1) / (len(diffs) + 1)

        result[f'diff_{metric_name}_ci_l'] = boot_result.confidence_interval.low
        result[f'diff_{metric_name}_ci_u'] = boot_result.confidence_interval.high
        result[f'diff_{metric_name}_se'] = boot_result.standard_error
        result[f'diff_{metric_name}_p'] = p_value

    return result


def apply_grouping(labels: np.ndarray, group_by) -> np.ndarray:
    if group_by == 'lang':
        return labels
    result = []
    for l in labels:
        parts = l.rsplit('_', 1)  # split from the RIGHT
        if len(parts) == 2 and parts[1].isalpha() and len(parts[1]) == 4:
            result.append(parts[1])
        else:
            result.append(l)  # fallback
    return np.array(result)

    
def split_by_method(group_by, embeddings, labels, method: str):
    mask = np.array([f'peft-{method}-' in str(l).lower() for l in labels])
    
    def extract_lang_script(label: str) -> str:
        parts = label.rsplit('@', 1)[0]
        lang_script = parts.rsplit('@', 1)[1]
        return lang_script
    
    clean = np.array([extract_lang_script(str(l)) for l in labels[mask]])
    clean = apply_grouping(clean, group_by)
    return embeddings[mask], clean


def process_file(hs_path: str, file_name: str, group_by, use_bootstrap=False, n_resamples=N_RESAMPLES) -> dict:
    all_emb, all_lab = extract_grouped_hs_embeddings(hs_path)

    xpe_emb, xpe_labels = split_by_method(group_by, all_emb, all_lab, 'xpe')
    spt_emb, spt_labels = split_by_method(group_by, all_emb, all_lab, 'spt')

    results = {
        'xpe': evaluate_clustering(xpe_emb, xpe_labels),
        'spt': evaluate_clustering(spt_emb, spt_labels),
    }

    row = {'file_name': file_name}
    for method in METHODS:
        for field in [*METRICS, *META]:
            row[f'{method}_{field}'] = results[method][field]
    for m in METRICS:
        row[f'diff_{m}'] = results['xpe'][m] - results['spt'][m]

    if use_bootstrap:
        boot = bootstrap_difference(xpe_emb, xpe_labels, spt_emb, spt_labels, n_resamples)
        row.update(boot)

    return row


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-name', type=str, default=DIR_NAME)
    parser.add_argument('--source-lang-group', type=str, default=SOURCE_LANG_GROUP, choices=['joshi5', 'seen'])
    parser.add_argument('--group-by', type=str, default=GROUP_BY, choices=['lang', 'script'])
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--n-resamples', type=int, default=N_RESAMPLES)
    args = parser.parse_args()

    in_path = f'{IN_PATH}/{args.dir_name}'
    boot_suffix = '_boot_diff_sci' if args.bootstrap else ''
    out_path = f'{OUT_PATH}/{args.dir_name}/hs_q_{args.source_lang_group}_{args.group_by}{boot_suffix}_res.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    csv_columns = get_csv_columns(use_bootstrap=args.bootstrap)

    if os.path.exists(out_path): os.remove(out_path)

    print(f'DIR_NAME={args.dir_name}')
    print(f'SOURCE_LANG_GROUP={args.source_lang_group}')
    print(f'GROUP_BY={args.group_by}')
    print(f'BOOTSTRAP={args.bootstrap}')
    if args.bootstrap:
        print(f'N_RESAMPLES={args.n_resamples}')

    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    for i in range(N_LAYERS):
        name = f'hs_q_{args.source_lang_group}_{i}'
        path = f'{in_path}/{name}.pt'
        if not os.path.exists(path):
            print(f'[SKIP] {path}')
            continue
        tasks.append((i, name, path))

    results = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_file, path, name, args.group_by, args.bootstrap, args.n_resamples): (i, name)
            for i, name, path in tasks
        }
        for future in as_completed(futures):
            i, name = futures[future]
            try:
                results[i] = future.result()
                print(f'[OK] {name} ({i + 1}/{N_LAYERS})')
            except Exception as e:
                print(f'[ERROR] {name}: {e}')

    for i, name, path in tasks:
        if i not in results:
            continue
        exists = os.path.exists(out_path)
        with open(out_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=csv_columns)
            if not exists:
                w.writeheader()
            w.writerow(results[i])


if __name__ == '__main__':
    main()