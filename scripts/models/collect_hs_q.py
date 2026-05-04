'''
Loads soft prompt models from Hugging Face and collects hidden states.

Config:
    xlmr/finetune/peft/sib200_collect.yml

Models (XPE / SPT soft prompts PEFT):
    mikaberidze/xlmr-large-sib200-peft-xpe-joshi5
    mikaberidze/xlmr-large-sib200-peft-spt-joshi5
    mikaberidze/xlmr-large-sib200-peft-xpe-seen 
    mikaberidze/xlmr-large-sib200-peft-spt-seen

Branches Per Model (Random Seeds): 
    seed-01, seed-02, seed-03 ... seed-10

Languages:
    seen:               tur_Latn, pes_Arab, ukr_Cyrl
    low-perf (unseen):  bem_Latn, ckb_Arab, tgk_Cyrl

Experiment:
    model use joshi5 XPEs and SPTs
    target langs use 5 seen and 5 low-performing languages
    collect hidden states for {0, 1, 2, 12, 13, 23, 24}
    store hidden states in a .pt file

Configs:
    xlmr/finetune/peft/sib200_collect
    xlmr/finetune/peft/sib200_from_hf.xpe
    
Usage:
runtime/clusters/pegasus/shell/run.sh "\
python -m micm_nlp.models.scripts.peft.xpe.collect_hs_q \
    --config xlmr/finetune/peft/sib200_collect \
    --batch-index 2 \
    --batch-size 5 \
    --random-pick 0.01 \
    --source-lang-group seen"

== ==
sbatch runtime/clusters/pegasus/shell/run.sh "\
python -m micm_nlp.models.scripts.peft.xpe.collect_hs_q \
    --config xlmr/finetune/peft/sib200_collect \
    --batch-index 3 \
    --batch-size 16 \
    --random-pick 0.05 \
    --source-lang-group seen"

sbatch runtime/clusters/pegasus/shell/run.sh "\
python -m micm_nlp.models.scripts.peft.xpe.collect_hs_q \
    --config xlmr/finetune/peft/sib200_collect \
    --batch-index 3 \
    --batch-size 16 \
    --random-pick 0.05 \
    --source-lang-group joshi5"

sbatch runtime/clusters/pegasus/shell/run.sh "\
python -m micm_nlp.models.scripts.peft.xpe.collect_hs_q \
    --config xlmr/finetune/peft/sib200_collect \
    --batch-index 4 \
    --batch-size 16 \
    --random-pick 0.05 \
    --source-lang-group seen"

sbatch runtime/clusters/pegasus/shell/run.sh "\
python -m micm_nlp.models.scripts.peft.xpe.collect_hs_q \
    --config xlmr/finetune/peft/sib200_collect \
    --batch-index 4 \
    --batch-size 16 \
    --random-pick 0.05 \
    --source-lang-group joshi5"

squeue -u bmikaberidze -l  
'''
import micm_nlp.utils as utils

import os
import copy
import torch
from collections import defaultdict
from experiments.config.xpe_utils import not_joshi5_sib200_ds_names, xlmr_unseen_sib200_ds_names
from micm_nlp.models.model import MODEL
from micm_nlp.trainer.trainer import TRAINER
from micm_nlp.enums import DsTypeSE
from experiments.models.peft.xpe.load import load_xpe_model

TEST_MODE = False

BATCH_SIZE = 5
BATCH_INDEX = 1
RANDOM_PICK = 0.05 # randomly pick some portion of hidden state tokens
SOURCE_LANG_GROUP = 'seen'

MODELS_BY_SOURCE_LANG_GROUPS = {
    'joshi5': [
        "mikaberidze/xlmr-large-sib200-peft-xpe-joshi5",
        "mikaberidze/xlmr-large-sib200-peft-spt-joshi5",
    ],
    'seen': [
        "mikaberidze/xlmr-large-sib200-peft-xpe-seen",
        "mikaberidze/xlmr-large-sib200-peft-spt-seen",
    ] 
}

TARGET_LANG_GROUP_BY_MODEL = {
    "mikaberidze/xlmr-large-sib200-peft-xpe-joshi5": "all_wo_j5",
    "mikaberidze/xlmr-large-sib200-peft-spt-joshi5": "all_wo_j5",
    "mikaberidze/xlmr-large-sib200-peft-xpe-seen": "unseen",
    "mikaberidze/xlmr-large-sib200-peft-spt-seen": "unseen",
}
TARGET_LANG_GROUPS = {
    "unseen": xlmr_unseen_sib200_ds_names,
    "all_wo_j5": not_joshi5_sib200_ds_names,
}

SEED_S = 1
SEED_E = 11
SEEDS = [f"seed-{i:02d}" for i in range(SEED_S, SEED_E)]

# HIDDEN_STATES_LAYERS = {0, 1, 2, 12, 13, 23, 24}
# HIDDEN_STATES_LAYERS = {3, 4, 5, 10, 14, 21, 22}
# HIDDEN_STATES_LAYERS = {6, 7, 8, 9, 15, 16, 17, 18, 19, 20}
HIDDEN_STATES_LAYERS = None

def get_batch_hidden_states(model, trainer):
    model._model.config.output_hidden_states = True
    model._model.config.return_dict = True
    model._model.eval()
    dataloader = trainer.trainer.get_train_dataloader()
    it = iter(dataloader)
    batch = None
    print(f"Getting batch {BATCH_INDEX} of {dataloader}")
    for i in range(BATCH_INDEX):
        batch = next(it)
    device = model._model.device
    if batch is None:
        raise ValueError(f"Batch index {BATCH_INDEX} is out of range")
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    with torch.no_grad():
        output = model._model(**batch)
    return output.hidden_states, batch['attention_mask']

# Print the key structure of "pf['prompt_hidden_states']"
def print_dict_tree(d, indent=0):
    if isinstance(d, dict):
        for k in d:
            print('  ' * indent + f"- {k}")
            print_dict_tree(d[k], indent + 1)
    elif isinstance(d, list):
        for idx, v in enumerate(d):
            print('  ' * indent + f"[{idx}]")
            print_dict_tree(v, indent + 1)
    else:
        # print the type or shape if it's a tensor
        if hasattr(d, 'shape'):
            print('  ' * indent + f"{type(d).__name__} shape={tuple(d.shape)}")
        else:
            print('  ' * indent + f"{type(d).__name__}")

def get_hidden_state_key(repo_id, group, language, layer):
    return f'{repo_id}@{group}@{language}@{layer}'

if __name__ == '__main__':
    from micm_nlp.config import CONFIG

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--batch-index', type=int, default=BATCH_INDEX)
    parser.add_argument('--random-pick', type=float, default=RANDOM_PICK)
    parser.add_argument('--source-lang-group', type=str, default=SOURCE_LANG_GROUP,
                        choices=list(MODELS_BY_SOURCE_LANG_GROUPS.keys()))
    args = parser.parse_args()

    config_name = args.config
    BATCH_SIZE = args.batch_size
    BATCH_INDEX = args.batch_index
    RANDOM_PICK = args.random_pick
    SOURCE_LANG_GROUP = args.source_lang_group
    HF_MODELS = MODELS_BY_SOURCE_LANG_GROUPS[SOURCE_LANG_GROUP]

    # Parse Config Name Argument
    # config_name = utils.parse_script_args()

    # Load base config once
    config = CONFIG.from_yaml(config_name)

    config.training_args.per_device_train_batch_size = BATCH_SIZE
    DS_DIRS = config.ds.dirs

    pf = {
        "meta": {
            "config_name": config_name,
            "models": HF_MODELS,
            "seeds": SEEDS,
            "language_groups": TARGET_LANG_GROUPS,
            "seed_unify": "concat_batch",
        },
        "prompt_hidden_states": {},
    }

    hs_by_layers = {}

    for repo_id in HF_MODELS:
        for seed in SEEDS:
            group = TARGET_LANG_GROUP_BY_MODEL[repo_id]
            for language in TARGET_LANG_GROUPS[group]:
                utils.p(f"\n[green]=== {repo_id} @ {seed} @ {group} @ {language} ===[/green]")

                if config.ds.type == DsTypeSE.HUGGINGFACE_SAVED:
                    config.ds.dirs = os.path.join(DS_DIRS, language)
                elif config.ds.type == DsTypeSE.HUGGINGFACE:
                    config.ds.name = language
                else:
                    raise ValueError(f"Unsupported dataset type: {config.ds.type}")

                model = load_xpe_model(config, repo_id, seed)
                trainer = TRAINER(model)
                utils.p(f"Model loaded: {model}")

                hidden_states, attention_mask = get_batch_hidden_states(model, trainer)
                prompt_size = config.task.peft.num_virtual_tokens

                for i, hs in enumerate(hidden_states):
                    if HIDDEN_STATES_LAYERS and i not in HIDDEN_STATES_LAYERS:
                        continue
                    if i not in hs_by_layers:
                        hs_by_layers[i] = defaultdict(list)

                    hs = hs.detach().cpu()
                    mask = attention_mask.detach().cpu()

                    # Prepend zeros for prompt positions (we want to exclude them)
                    prompt_pad = torch.zeros(mask.shape[0], prompt_size, dtype=mask.dtype)
                    mask = torch.cat([prompt_pad, mask], dim=1)  # now (batch_size, prompt_size + input_seq_len)

                    # Flatten and filter
                    hs_flat = hs.reshape(-1, hs.shape[-1])
                    mask_flat = mask.reshape(-1).bool()
                    hs_filtered = hs_flat[mask_flat]

                    if RANDOM_PICK:
                        n = int(hs_filtered.shape[0] * RANDOM_PICK)
                        hs_filtered = hs_filtered[torch.randperm(hs_filtered.shape[0])[:n]]

                    key = get_hidden_state_key(repo_id, group, language, i)
                    hs_by_layers[i][key].append(hs_filtered)

                del trainer, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if TEST_MODE: break
            if TEST_MODE: break
        if TEST_MODE: break

    for layer, hs_by_key in hs_by_layers.items():
        pf["prompt_hidden_states"] = {
            key: torch.cat(tensors, dim=0)
            for key, tensors in hs_by_key.items()
        }

        print("\n[bold yellow]PEFT prompt_hidden_states key structure:[/bold yellow]")
        print_dict_tree(pf['prompt_hidden_states'])
        # if TEST_MODE: exit()

        output_name = f"hs_q_{SOURCE_LANG_GROUP}_{layer}.pt"
        output_dir = os.path.join(MODEL.stor_path, "xlmr", "FacebookAI|xlm-roberta-large", "prompt_hidden_states", f"batch_{BATCH_INDEX}_{BATCH_SIZE}_rand_{int(RANDOM_PICK*100)}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        torch.save(pf, output_path)
        utils.p(f"\n[bold green]Saved prompt hidden states to {output_path}[/bold green]")
        # /home/bmikaberidze/XPE/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/prompt_hidden_states/prompt_hidden_states.pt


