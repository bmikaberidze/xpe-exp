"""
Usage:

sbatch runtime/clusters/pegasus/shell/run.sh --no-gpu "\
python -m scripts.evals.quant_boot_sci \
    --dir-name batch_3_16_rand_5 \
    --source-lang-group joshi5 \
    --group-by lang \
    --bootstrap \
    --n-resamples 1000 \
    --worker 64 \
"& \
sbatch runtime/clusters/pegasus/shell/run.sh --no-gpu "\
python -m scripts.evals.quant_boot_sci \
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
        for method in METHODS:
            for metric in BOOT_METRICS:
                cols.append(f'{method}_{metric}_ci_l')
                cols.append(f'{method}_{metric}_ci_u')
                cols.append(f'{method}_{metric}_se')

    return cols


# ── Core ────────────────────────────────────────────────────────────────────────
def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_resamples = 0 # non-zero runs Bootstrap
) -> dict:

    unique = np.unique(labels)
    numeric = np.array([{l: i for i, l in enumerate(unique)}[l] for l in labels])

    sil = 0.0
    if not n_resamples:
        if SIL_N_SAMPLES is None:
            sil = silhouette_score(embeddings, numeric, metric=SIL_DIST_METRIC)
        elif SIL_N_SAMPLES:
            idx = resample(np.arange(len(embeddings)), n_samples=SIL_N_SAMPLES, random_state=42)
            sil = silhouette_score(embeddings[idx], numeric[idx], metric=SIL_DIST_METRIC)

    result = {
        'sil': sil,
        'db': davies_bouldin_score(embeddings, numeric),
        'ch': calinski_harabasz_score(embeddings, numeric),
        'n_samples': len(embeddings),
        'n_clusters': len(unique),
    }

    if n_resamples:
        indices = (np.arange(len(embeddings)),)
        for metric_name, metric_fn in [('db', davies_bouldin_score),
                                        ('ch', calinski_harabasz_score)]:
            def statistic(idx, 
                          _emb=embeddings, _num=numeric, 
                          _fn=metric_fn):
                idx = idx.astype(int)
                return _fn(_emb[idx], _num[idx])

            boot_result = scipy_bootstrap(
                indices,
                statistic,
                n_resamples=n_resamples,
                confidence_level=0.95,
                method='percentile',
                random_state=42,
            )
            print(boot_result)
            exit()
            # BootstrapResult(
            #     confidence_interval=ConfidenceInterval(low=8.724908374071937, high=8.790272141831796), 
            #     bootstrap_distribution=array([8.77183612, 8.75420639, 8.78008212, 8.74869254, 8.75316348, 8.76532584, 8.72331011, 8.77710945, 8.79323053, 8.73041349]), 
            #     standard_error=0.02210513203869379)

            result[f'{metric_name}_ci_l'] = boot_result.confidence_interval.low
            result[f'{metric_name}_ci_u'] = boot_result.confidence_interval.high
            result[f'{metric_name}_se'] = boot_result.standard_error


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

    if use_bootstrap:
        results = {m: evaluate_clustering(*split_by_method(group_by, all_emb, all_lab, m), n_resamples=n_resamples) for m in METHODS}
    else:
        results = {m: evaluate_clustering(*split_by_method(group_by, all_emb, all_lab, m)) for m in METHODS}

    row = {'file_name': file_name}
    for method in METHODS:
        for field in [*METRICS, *META]:
            row[f'{method}_{field}'] = results[method][field]
    for m in METRICS:
        row[f'diff_{m}'] = results['xpe'][m] - results['spt'][m]

    if use_bootstrap:
        for method in METHODS:
            for metric in BOOT_METRICS:
                row[f'{method}_{metric}_ci_l'] = results[method][f'{metric}_ci_l']
                row[f'{method}_{metric}_ci_u'] = results[method][f'{metric}_ci_u']
                row[f'{method}_{metric}_se'] = results[method][f'{metric}_se']

    return row


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-name', type=str, default=DIR_NAME)
    parser.add_argument('--source-lang-group', type=str, default=SOURCE_LANG_GROUP, choices=['joshi5', 'seen'])
    parser.add_argument('--group-by', type=str, default=GROUP_BY, choices=['lang', 'script'])
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--n-resamples', type=int, default=N_RESAMPLES)
    args = parser.parse_args()

    in_path = f'{IN_PATH}/{args.dir_name}'
    boot_suffix = '_boot_sci' if args.bootstrap else ''
    out_path = f'{OUT_PATH}/{args.dir_name}/hs_q_{args.source_lang_group}_{args.group_by}{boot_suffix}_res.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    csv_columns = get_csv_columns(use_bootstrap=args.bootstrap)

    resume_from = -1
    if os.path.exists(out_path):
        if not args.resume:
            os.remove(out_path)
        else:
            from io import StringIO
            with open(out_path, 'r') as f:
                lines = f.readlines()
            if len(lines) > 1:
                last_row = next(csv.reader(StringIO(lines[-1])))
                last_layer = last_row[0].split('_')[-1]
                if last_layer.isdigit():
                    resume_from = int(last_layer)
                    print(f'[RESUME] Resuming after layer {resume_from}')
            else:
                print('[RESUME] Only header found, starting from scratch')

    print(f'DIR_NAME={args.dir_name}')
    print(f'SOURCE_LANG_GROUP={args.source_lang_group}')
    print(f'GROUP_BY={args.group_by}')
    print(f'BOOTSTRAP={args.bootstrap}')
    if args.bootstrap:
        print(f'N_RESAMPLES={args.n_resamples}')

    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    for i in range(N_LAYERS):
        if i <= resume_from:
            print(f'[SKIP] Layer {i} already processed')
            continue
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