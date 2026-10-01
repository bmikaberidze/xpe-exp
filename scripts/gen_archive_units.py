"""Rewrite the old decoder grids (config/meta/*) as micm-nlp 0.5 group configs whose
units reproduce what each run ACTUALLY used (its wandb config.yaml).

Writes:
  config/units/archive/{grid}/{tune.<method>|test}.lm.<llm>.ds.bebe[.fold][.vN].yml
  config/groups/{grid}.{source_group}.yml
Then re-checks: archive unit + entry overrides == wandb record, for every run.

Run (container): python scripts/gen_archive_units.py [--write]
It reads the old metaconfigs from artefacts/_scratch/refactor/meta/ -- config/meta/ was deleted
right after this ran; recreate with:
  for f in $(git ls-tree --name-only 44ac0cb config/meta/); do git show 44ac0cb:$f > artefacts/_scratch/refactor/meta/$(basename $f); done
One-off (2026-09-30): it rewrote grids 8-16; kept for the provenance of config/units/archive/.
"""
import copy
import glob
import os
import re
import sys
from collections import Counter, defaultdict

import yaml

ROOT = '/fscratch/bmikaberidze/xpe-exp'
WRITE = '--write' in sys.argv
GRIDS = ['8', '9a', '9b', '9c', '10a', '10b', '11', '12', '13a', '13b', '14b', '15a', '16']
LLM_FILE = {'aya': 'aya', 'bloom': 'bloomz-7b1'}          # xlt_runs dir -> unit file tag
SRC_ALIAS = {'arb_Arab-eng_Latn-fra_Latn-hin_Deva-ind_Latn-spa_Latn-zho_Hans': 'self_split7'}
SECTIONS = ['mode', 'ds', 'eval', 'test', 'custom_training_args', 'data_collator', 'model',
            'peft', 'task', 'tokenizer', 'trainer', 'training_args']
IGNORE = {
    'ds.dirs', 'ds.path', 'eval.after_training', 'test.save_predictions', 'tokenizer.vocab_size',
    'model.uuid4', 'model.param_size', 'model.trainable_param_size',
    'model.trainable_param_size_ratio', 'model.pretrained.adapter', 'model.pretrained.uuid4',
    'model.pretrained.args', 'training_args.args.seed', 'training_args.args.metric_for_best_model',
    'training_args.args.save_safetensors', 'training_args.args.overwrite_output_dir',
    'training_args.args.output_dir', 'training_args.args.run_name', 'training_args.args.logging_dir',
}
RENAMES = {'ds.comes_with_splits': 'ds.splits',
           'training_args.args.warmup_ratio': 'training_args.args.warmup_steps'}
ABSENT = object()
OVERRIDE_DEFAULTS = {'training_args.args.label_smoothing_factor': 0.0}   # HF TrainingArguments default
GRID_NOTES = {
    '15a_stab_probe_aya': (
        '# !! The *_clip_* entries NEVER normalised in the old runs: the callback was not\n'
        '#    registered before micm-nlp d3ae7e1 (2026-08-11), so they equalled *_base_*.\n'
        '#    In 0.5 the override IS applied -- a rerun of them is a new experiment.\n'
        '#    The *_ls_* entries crashed in the old code (label_smoother vs virtual tokens).\n'
        '#\n'),
    '8_bebe_self_split_bloomz': (
        '# Source langs = the 7 passed as --source-langs in the old dispatch, registered as\n'
        '# LANG_GROUPS["self_split7"]. The zero_shot entry has no old run dir.\n'
        '#\n'),
}


# ---------------------------------------------------------------- helpers
def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        p = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flat(v, p))
        else:
            out[p] = v
    return out


def parse_optim_args(s):
    out = {}
    for part in s.split(','):
        k, v = part.strip().split('=')
        v = {'None': None, 'True': True, 'False': False}.get(v, v)
        if isinstance(v, str):
            v = float(v)
        out[k] = v
    return out


def view(cfg):
    """Flat {dotted: raw value} over the micm sections, old names mapped to 0.5 ones."""
    f = {}
    for s in SECTIONS:
        v = cfg.get(s)
        if isinstance(v, dict):
            f.update(flat(v, s))
        elif v is not None:
            f[s] = v
    out = {}
    for k, v in f.items():
        for old, new in RENAMES.items():
            if k == old or k.startswith(old + '.'):
                k = new + k[len(old):]
        if k in IGNORE or any(k.startswith(i + '.') for i in IGNORE):
            continue
        if k == 'mode' and isinstance(v, str):
            v = v.lower()
        if k == 'training_args.args.group_by_length':
            k, v = 'training_args.args.train_sampling_strategy', ('group_by_length' if v else None)
        if k == 'training_args.args.optim_args' and isinstance(v, str):
            for kk, vv in parse_optim_args(v).items():
                out[f'{k}.{kk}'] = vv
            continue
        out[k] = v
    return out


def same(a, b):
    if a is ABSENT and b is None or b is ABSENT and a is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def drift(ran, unit):
    """{key: ran value (ABSENT = remove)} for every key where the run differs."""
    def eq(k, a, b):
        if k in OVERRIDE_DEFAULTS and {repr(a), repr(b)} <= {repr(ABSENT), repr(OVERRIDE_DEFAULTS[k])}:
            return True     # absent == the library default for keys the generator fills in
        return same(a, b)
    return {k: ran.get(k, ABSENT) for k in sorted(set(ran) | set(unit))
            if not eq(k, ran.get(k, ABSENT), unit.get(k, ABSENT))}


def set_dotted(cfg, key, value):
    parts = key.split('.')
    d = cfg
    for p in parts[:-1]:
        if not isinstance(d.get(p), dict):
            d[p] = {}
        d = d[p]
    if value is ABSENT:
        # absent in the run: the 0.5 key that replaced it defaults to None semantics
        d[parts[-1]] = None
    else:
        d[parts[-1]] = value


def unwrap(w):
    return {k: (v['value'] if isinstance(v, dict) and 'value' in v else v) for k, v in w.items()}


def fold_of(dirs):
    m = re.search(r'/fold(\d)/', dirs or '')
    return int(m.group(1)) if m else None


def fmt(v):
    return '<absent>' if v is ABSENT else repr(v)


# ---------------------------------------------------------------- wandb index
tune_idx, test_idx = {}, {}
for p in glob.glob(f'{ROOT}/artefacts/wandb/run-*/files/config.yaml'):
    txt = open(p).read()
    i = txt.find('\nmodel:\n    value:\n')
    if i < 0:
        continue
    block = []
    for line in txt[i + len('\nmodel:\n    value:\n'):].split('\n'):
        if not line.startswith('        '):
            break
        block.append(line)
    in_pre = False
    for line in block:
        if line == '        pretrained:':
            in_pre = True
            continue
        if not line.startswith('            '):
            in_pre = False
        m = re.match(r'^        uuid4: ([0-9a-f-]{36})\s*$', line)
        if m:
            tune_idx[m.group(1)] = p
        m = re.match(r'^            adapter: ([0-9a-f-]{36})\s*$', line)
        if in_pre and m:
            test_idx.setdefault(m.group(1), p)
print(f'# wandb: {len(tune_idx)} tune configs by model.uuid4, {len(test_idx)} test configs by adapter')

problems = []

for g in GRIDS:
    meta_path = glob.glob(f'{ROOT}/artefacts/_scratch/refactor/meta/{g}_*.yml')[0]  # config/meta/ as of 44ac0cb (see docstring)
    grid = os.path.basename(meta_path)[:-4]
    meta = yaml.safe_load(open(meta_path))
    tune_cfgs = meta['tune_configs']
    matrix = meta['matrix']
    by_name = {e['run_name']: e for e in matrix}

    # ---- collect runs
    runs = []   # dicts: src, llm, run_name, fold, entry, tune_view, test_view, tested
    for td in sorted(glob.glob(f'{ROOT}/artefacts/evals/xlt_runs/*/*/{grid}/*/')):
        parts = td.rstrip('/').split('/')
        llm, src, run_dir = parts[-4], parts[-3], parts[-1]
        run_name = '_'.join(run_dir.split('_')[2:])
        e = by_name.get(run_name)
        tune_yml = os.path.join(td, 'tune.yml')
        if e is None or not os.path.exists(tune_yml):
            problems.append(f'{grid}: {src}/{run_dir}: no matrix entry or tune.yml -> skipped')
            continue
        t = yaml.safe_load(open(tune_yml))
        uuid = t.get('adapter_uuid4')
        wp = tune_idx.get(uuid)
        if wp is None:
            problems.append(f'{grid}: {src}/{run_dir}: no wandb for adapter {uuid} -> skipped')
            continue
        w = unwrap(yaml.safe_load(open(wp)))
        tested = os.path.exists(os.path.join(td, 'raw.csv'))
        test_w = None
        if tested and uuid in test_idx:
            test_w = unwrap(yaml.safe_load(open(test_idx[uuid])))
        runs.append(dict(src=SRC_ALIAS.get(src, src), llm=llm, run_name=run_name,
                         fold=fold_of(w['ds'].get('dirs')), entry=e, w=w, test_w=test_w,
                         tested=tested, dir=f'{src}/{run_dir}'))
    if not runs:
        problems.append(f'{grid}: no runs found')
        continue
    llm = runs[0]['llm']
    lf = LLM_FILE[llm]

    # ---- tune variants: per method, distinct drift vs current unit (+ entry overrides removed)
    variants = {}        # (method, drift-tuple) -> key
    variant_files = {}   # key -> (filename, unit dict, drift, runs)
    run_variant = {}
    for r in runs:
        method = r['entry']['tune_method']
        unit_path = f'{ROOT}/config/units/' + os.path.basename(tune_cfgs[method])
        unit = yaml.safe_load(open(unit_path))
        with_ov = copy.deepcopy(unit)
        for k, v in (r['entry'].get('overrides') or {}).items():
            set_dotted(with_ov, k, v)
        d = drift(view(r['w']), view(with_ov))
        sig = (method, tuple((k, fmt(v)) for k, v in d.items()))
        if sig not in variants:
            variants[sig] = None
            variant_files.setdefault(method, []).append((sig, unit_path, unit, d))
        run_variant[id(r)] = sig
    sig_key = {}
    for method, lst in variant_files.items():
        cfg_method = 'dual' if 'dual' in os.path.basename(tune_cfgs[method]) else method
        for i, (sig, unit_path, unit, d) in enumerate(lst):
            suffix = '' if len(lst) == 1 else f'.v{i + 1}'
            key = method if len(lst) == 1 else f'{method}_v{i + 1}'
            fname = f'tune.{cfg_method}.lm.{lf}.ds.bebe{suffix}.yml'
            if method != cfg_method:            # e.g. 14b d30/d70 -> both use tune.dual
                fname = f'tune.{cfg_method}.{method}.lm.{lf}.ds.bebe{suffix}.yml'
            snap = copy.deepcopy(unit)
            # overrides stay in the entries, so drift keys the overrides own are not baked in
            for k, v in d.items():
                set_dotted(snap, k, v)
            # 0.5 overrides are strict (unknown key raises); the old CLI created keys. Give
            # every key an entry overrides its library default, so the base arms are unchanged.
            snap_view = view(snap)
            for e in matrix:
                if e.get('tune_method') != method:
                    continue
                for k in (e.get('overrides') or {}):
                    if k not in snap_view:
                        if k not in OVERRIDE_DEFAULTS:
                            raise SystemExit(f'{grid}: override {k} not in the unit and no known default')
                        set_dotted(snap, k, OVERRIDE_DEFAULTS[k])
                        d[k] = OVERRIDE_DEFAULTS[k]
                        snap_view[k] = OVERRIDE_DEFAULTS[k]
            sig_key[sig] = (key, fname, snap, d, unit_path)

    # ---- test variant (one per grid; tested runs only)
    test_cfg_name = f'test.lm.{lf}.ds.bebe.fold.yml'
    test_unit_path = f'{ROOT}/config/units/{test_cfg_name}'
    test_unit = yaml.safe_load(open(test_unit_path))
    test_sigs = Counter()
    test_drift = {}
    for r in runs:
        if r['test_w'] is not None:
            d = drift(view(r['test_w']), view(test_unit))
            s = tuple((k, fmt(v)) for k, v in d.items())
            test_sigs[s] += 1
            test_drift[s] = d
        elif r['tested']:
            problems.append(f"{grid}: {r['dir']}: tested but no test wandb -> test drift unchecked")
    test_snap = None
    if test_sigs:
        if len(test_sigs) > 1:
            problems.append(f'{grid}: {len(test_sigs)} distinct TEST drifts {list(test_sigs.values())} -> using the most common')
        s = test_sigs.most_common(1)[0][0]
        test_snap = copy.deepcopy(test_unit)
        for k, v in test_drift[s].items():
            set_dotted(test_snap, k, v)
        test_d = test_drift[s]

    # ---- report
    print(f'\n########## {grid}  llm={llm}  runs={len(runs)}  sources={sorted({r["src"] for r in runs})}')
    for sig, (key, fname, snap, d, unit_path) in sig_key.items():
        n = sum(1 for r in runs if run_variant[id(r)] == sig)
        folds = sorted({r['fold'] for r in runs if run_variant[id(r)] == sig})
        print(f'  [{key}] {fname}  <- {os.path.basename(unit_path)}  ({n} runs, folds {folds})')
        for k, v in d.items():
            print(f'      {k}: {fmt(v)}')
    if test_snap is not None:
        print(f'  [test] {test_cfg_name}  ({sum(test_sigs.values())} tested runs)')
        for k, v in test_d.items():
            print(f'      {k}: {fmt(v)}')

    # ---- entries: matrix x dispatched (source, fold)
    dispatched = sorted({(r['src'], r['fold']) for r in runs}, key=lambda x: (x[0], -1 if x[1] is None else x[1]))
    by_src = defaultdict(list)
    for src, fold in dispatched:
        for e in matrix:
            method = e.get('tune_method')
            if e.get('fold') is not None and e['fold'] != fold:
                continue
            match = [r for r in runs if r['src'] == src and r['fold'] == fold and r['run_name'] == e['run_name']]
            if method is None:                                   # zero-shot entry
                entry = {'config': 'test', 'name': f"{e['run_name']}_f{fold}", 'fold': fold,
                         'method': 'zero', 'source_group': src, 'llm': llm}
                by_src[src].append((entry, 'not run in the old grid' if not match else ''))
                continue
            if match:
                # one entry per distinct recipe this (entry, fold) ran with -- a re-dispatch
                # under a changed config (9a: restarts, then plain cosine) is its own entry
                seen = {}
                for r in sorted(match, key=lambda r: r['dir']):
                    seen.setdefault(run_variant[id(r)], r)
                todo = [(s, r['tested'], f"old run {r['dir'].split('/')[-1][:15]}") for s, r in seen.items()]
            else:  # never ran (crashed / not dispatched): method's most common variant
                c = Counter(run_variant[id(r)] for r in runs if r['entry'].get('tune_method') == method)
                todo = [(c.most_common(1)[0][0], any(r['tested'] for r in runs), 'no run dir in the old grid')]
            for sig, tested, note in todo:
                key = sig_key[sig][0]
                name = e['run_name'] if re.search(r'_f\d', e['run_name']) else f"{e['run_name']}_f{fold}"
                if len(todo) > 1:
                    name += '_' + key.split('_')[-1]
                entry = {'config': key, 'name': name}
                if e.get('seed') is not None:
                    entry['seed'] = e['seed']
                entry.update({'fold': fold, 'method': method, 'source_group': src, 'llm': llm})
                if e.get('overrides'):
                    entry['overrides'] = e['overrides']
                if tested:
                    entry['separate_test'] = {'config': 'test'}
                else:
                    entry['tune_only'] = True
                by_src[src].append((entry, note if len(todo) > 1 or not match else ''))

    # ---- self-check: snapshot + overrides == wandb, per run
    bad = 0
    for r in runs:
        key, fname, snap, d, _ = sig_key[run_variant[id(r)]]
        chk = copy.deepcopy(snap)
        for k, v in (r['entry'].get('overrides') or {}).items():
            set_dotted(chk, k, v)
        rest = drift(view(r['w']), view(chk))
        if rest:
            bad += 1
            problems.append(f"{grid}: {r['dir']}: residual drift {list(rest)}")
    print(f'  self-check: {len(runs) - bad}/{len(runs)} runs reproduce their wandb config')

    if not WRITE:
        continue

    # ---- write archive units
    out_dir = f'{ROOT}/config/units/archive/{grid}'
    os.makedirs(out_dir, exist_ok=True)
    for sig, (key, fname, snap, d, unit_path) in sig_key.items():
        folds = sorted({r['fold'] for r in runs if run_variant[id(r)] == sig})
        hdr = [f'# ARCHIVE unit for grid {grid} ({key}). GENERATED -- do not edit.',
               f'# = config/units/{os.path.basename(unit_path)} + the keys below, which is what',
               f'#   the old runs actually used (their wandb config.yaml; folds {folds}).',
               '# Generator: scripts/gen_archive_units.py (2026-09-30).',
               '# Differences from the current unit:']
        hdr += [f'#   {k}: {fmt(v)}' for k, v in d.items()] or ['#   (none)']
        with open(os.path.join(out_dir, fname), 'w') as fh:
            fh.write('\n'.join(hdr) + '\n\n')
            yaml.safe_dump(snap, fh, sort_keys=False, allow_unicode=True, width=100)
    if test_snap is not None:
        raise SystemExit(f'{grid}: test wandb found -- snapshot writing for it is not implemented')
    if any(r['tested'] for r in runs) or any(e.get('tune_method') is None for e in matrix):
        with open(os.path.join(out_dir, test_cfg_name), 'w') as fh:
            fh.write(f'# ARCHIVE test unit for grid {grid}: a verbatim copy of config/units/{test_cfg_name}\n'
                     '# (2026-09-30). The old test phase was not logged to wandb, so it cannot be checked\n'
                     '# against the runs; git shows the file unchanged since its first commit (bloomz\n'
                     '# 37292a0 06-10, aya a0d6c1e 06-24) apart from the 0.4/0.5 key migrations. Grids 8\n'
                     '# and 12 ran BEFORE that first commit (uncommitted file): their test setup is unverified.\n\n')
            fh.write(open(test_unit_path).read())

    # ---- write group configs, one per source group
    for src, entries in by_src.items():
        keys_used = {e['config'] for e, _ in entries} | {e['separate_test']['config'] for e, _ in entries if 'separate_test' in e}
        configs = {}
        for sig, (key, fname, *_rest) in sig_key.items():
            if key in keys_used:
                configs[key] = f'../units/archive/{grid}/{fname}'
        if 'test' in keys_used:
            configs['test'] = f'../units/archive/{grid}/{test_cfg_name}'
        path = f'{ROOT}/config/groups/{grid}.{src}.yml'
        with open(path, 'w') as fh:
            fh.write(f'# {grid}, source group {src.upper()} ({llm}) -- rewritten from config/meta/{grid}.yml\n'
                     f'# (tag pre-micm-nlp-0.4) for micm-nlp 0.5. GENERATED by\n'
                     f'# scripts/gen_archive_units.py (2026-09-30).\n'
                     f'#\n'
                     f'# Units are ARCHIVE snapshots: what the old runs actually used (wandb), so a\n'
                     f'# rerun is the same experiment. Entries = old matrix x the folds it was\n'
                     f'# dispatched on. `tune_only: true` = the old dispatch used --skip-test.\n'
                     f'# Results under artefacts/evals/xlt_runs/{llm}/<src>/{grid}/ came from the\n'
                     f'# pre-0.4 code; a rerun here is a NEW result (new stack), not a replacement.\n'
                     f'#\n'
                     + GRID_NOTES.get(grid, '') +
                     f'# Dispatch:\n'
                     f'#   sbatch --array=0-{len(entries) - 1}%10 --mem=80G \\\n'
                     f'#     --partition={"B200,H200,H200-PCI,H100-PCI" if llm == "aya" else "B200,H200,H200-PCI,H100-PCI,H100,A100-80GB"} \\\n'
                     f'#     runtime/clusters/pegasus/shell/run.sh --site-packages \\\n'
                     f'#       "python -m micm_nlp run-group \\\n'
                     f'#          --group-config config/groups/{grid}.{src}.yml \\\n'
                     f'#          --runner scripts.xlt_runner:run \\\n'
                     f'#          --root-path /fscratch/bmikaberidze/xpe-exp"\n\n')
            fh.write('configs:\n')
            for k, v in configs.items():
                fh.write(f'  {k}: {v}\n')
            fh.write('\nruns:\n')
            for e, note in entries:
                line = yaml.safe_dump(e, default_flow_style=True, sort_keys=False, width=10_000).strip()
                fh.write(f'  - {line}' + (f'   # {note}' if note else '') + '\n')

print('\n########## PROBLEMS')
for p in problems:
    print('  ' + p)
