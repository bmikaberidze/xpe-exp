"""
Usage:

sbatch runtime/clusters/pegasus/shell/run.sh --no-gpu "\
python -m scripts.evals.quant\
    --dir-name batch_3_16_rand_5 \
    --source-lang-group joshi5 \
    --group-by lang \
    --resume \
" & \
sbatch runtime/clusters/pegasus/shell/run.sh --no-gpu "\
python -m scripts.evals.quant\
    --dir-name batch_4_16_rand_5 \
    --source-lang-group joshi5 \
    --group-by lang \
    --resume \
"

squeue -u bmikaberidze -l  
"""

import csv
import os
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from micm_nlp.models.model import MODEL
from scripts.evals.utils import extract_grouped_hs_embeddings

# ── Config ──────────────────────────────────────────────────────────────────────
GROUP_BY = 'script'  # 'lang' | 'script'
DIR_NAME = 'batch_1_5_rand_1'
SOURCE_LANG_GROUP = 'joshi5'
# 
N_LAYERS = 25
SIL_N_SAMPLES = None  # None = full, 0 = skip, int = subsample
SIL_DIST_METRIC = 'manhattan'
# 
IN_PATH = f'{MODEL.stor_path}/xlmr/FacebookAI|xlm-roberta-large/prompt_hidden_states'
OUT_PATH = 'artefacts/evals/hs_quant'
# 
METHODS = ('xpe', 'spt')
METRICS = ('silhouette', 'davies_bouldin', 'calinski_harabasz')
META = ('n_samples', 'n_clusters')
# 
CSV_COLUMNS = ['file_name'] + [
    f'{method}_{field}'
    for field in [*METRICS, *META]
    for method in METHODS
] + [f'diff_{m}' for m in METRICS]


# ── Core ────────────────────────────────────────────────────────────────────────
def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    unique = np.unique(labels)
    numeric = np.array([{l: i for i, l in enumerate(unique)}[l] for l in labels])

    if SIL_N_SAMPLES is None:
        sil = silhouette_score(embeddings, numeric, metric=SIL_DIST_METRIC)
    elif SIL_N_SAMPLES:
        idx = resample(np.arange(len(embeddings)), n_samples=SIL_N_SAMPLES, random_state=42)
        sil = silhouette_score(embeddings[idx], numeric[idx], metric=SIL_DIST_METRIC)
    else:
        sil = 0.0

    return {
        'silhouette': sil,
        'davies_bouldin': davies_bouldin_score(embeddings, numeric),
        'calinski_harabasz': calinski_harabasz_score(embeddings, numeric),
        'n_samples': len(embeddings),
        'n_clusters': len(unique),
    }


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
        # Extract "ace_Arab" from "...@unseen@ace_Arab@0"
        parts = label.rsplit('@', 1)[0]  # drop trailing @0
        lang_script = parts.rsplit('@', 1)[1]  # get ace_Arab
        return lang_script
    
    clean = np.array([extract_lang_script(str(l)) for l in labels[mask]])
    clean = apply_grouping(clean, group_by)
    return embeddings[mask], clean


def process_file(hs_path: str, file_name: str, group_by) -> dict:
    all_emb, all_lab = extract_grouped_hs_embeddings(hs_path)
    # print(np.unique(all_lab)[:10])
    # exit()
    results = {m: evaluate_clustering(*split_by_method(group_by, all_emb, all_lab, m)) for m in METHODS}

    row = {'file_name': file_name}
    for method in METHODS:
        for field in [*METRICS, *META]:
            row[f'{method}_{field}'] = results[method][field]
    for m in METRICS:
        row[f'diff_{m}'] = results['xpe'][m] - results['spt'][m]
    return row


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-name', type=str, default=DIR_NAME)
    parser.add_argument('--source-lang-group', type=str, default=SOURCE_LANG_GROUP, choices=['joshi5', 'seen'])
    parser.add_argument('--group-by', type=str, default=GROUP_BY, choices=['lang', 'script'])
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    in_path = f'{IN_PATH}/{args.dir_name}'
    out_path = f'{OUT_PATH}/{args.dir_name}/hs_q_{args.source_lang_group}_{args.group_by}_res.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    resume_from = -1
    if os.path.exists(out_path):
        if not args.resume:
            os.remove(out_path)
        else:
            from io import StringIO
            with open(out_path, 'r') as f:
                lines = f.readlines()
            if len(lines) > 1:  # more than just header
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

    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Build list of layers to process (respecting resume)
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

    results = {}  # i -> row

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_file, path, name, args.group_by): (i, name)
            for i, name, path in tasks
        }
        for future in as_completed(futures):
            i, name = futures[future]
            try:
                results[i] = future.result()
                print(f'[OK] {name} ({i + 1}/{N_LAYERS})')
            except Exception as e:
                print(f'[ERROR] {name}: {e}')

    # Write results in layer order
    for i, name, path in tasks:
        if i not in results:
            continue
        exists = os.path.exists(out_path)
        with open(out_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not exists:
                w.writeheader()
            w.writerow(results[i])


if __name__ == '__main__':
    main()