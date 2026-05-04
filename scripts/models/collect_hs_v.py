'''
Loads soft prompt models from Hugging Face and collects hidden states.

Config:
    xlmr/finetune/peft/sib200_from_hf.xpe.yml

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

Usage:
    python -m micm_nlp.models.scripts.peft.xpe.collect_hs_v \
        --config xlmr/finetune/peft/sib200_from_hf.xpe

'''
import micm_nlp.utils as utils

import os
import copy
import torch
from collections import defaultdict
from micm_nlp.models.model import MODEL
from micm_nlp.trainer.trainer import TRAINER
from experiments.models.peft.xpe.load import load_xpe_model

TEST_MODE = False

seen_and_lowperf_sib200_scripts = [
    'Arab',
    'Beng',
    'Cyrl',
    'Latn',
    'Mymr',
]
lang_by_script_and_groups = {
    'lowperf': {
        'Arab': ['knc_Arab', 'ckb_Arab', 'min_Arab', 'bjn_Arab', 'ace_Arab'],
        'Beng': ['mni_Beng'],
        'Cyrl': ['tgk_Cyrl'],
        'Latn': ['umb_Latn', 'kam_Latn', 'nus_Latn', 'run_Latn', 'bem_Latn'],
        'Mymr': ['shn_Mymr'],
    },
    'seen': {
        'Arab': ['pes_Arab', 'urd_Arab', 'uig_Arab', 'azb_Arab', 'snd_Arab'],
        'Beng': ['ben_Beng'],
        'Cyrl': ['ukr_Cyrl'],
        'Latn': ['tur_Latn', 'eus_Latn', 'cat_Latn', 'fin_Latn', 'vie_Latn'],
        'Mymr': ['mya_Mymr'],
    }
}

LANGUAGE_GROUPS = {
    # "seen": ["tur_Latn", "pes_Arab", "ukr_Cyrl"],
    # "low-perf": ["bem_Latn", "ckb_Arab", "tgk_Cyrl"],
    "seen": [item for sublist in lang_by_script_and_groups["seen"].values() for item in sublist],
    "low-perf": [item for sublist in lang_by_script_and_groups["lowperf"].values() for item in sublist],
}

HF_MODELS = [
    "mikaberidze/xlmr-large-sib200-peft-xpe-joshi5",
    "mikaberidze/xlmr-large-sib200-peft-spt-joshi5",
    # "mikaberidze/xlmr-large-sib200-peft-xpe-seen",
    # "mikaberidze/xlmr-large-sib200-peft-spt-seen",
]

SEEDS = [f"seed-{i:02d}" for i in range(1, 11)]

HIDDEN_STATES_LAYERS = {0, 1, 2, 12, 13, 23, 24}

RANDOM_PICK = 0.1 # randomly pick some portion of hidden state tokens

def first_batch_hidden_states(model, trainer):
    model._model.config.output_hidden_states = True
    model._model.config.return_dict = True
    model._model.eval()
    dataloader = trainer.trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    device = model._model.device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    with torch.no_grad():
        output = model._model(**batch)
    return output.hidden_states

def get_hidden_state_key(repo_id, group, language, layer):
    return f'{repo_id}@{group}@{language}@{layer}'

if __name__ == '__main__':
    from micm_nlp.config import CONFIG

    # Parse Config Path Argument
    config_path = utils.parse_script_args()

    # Load base config once
    config = CONFIG.from_yaml(config_path)

    pf = {
        "meta": {
            "config_name": config_name,
            "models": HF_MODELS,
            "seeds": SEEDS,
            "language_groups": LANGUAGE_GROUPS,
            "seed_unify": "concat_batch",
        },
        "prompt_hidden_states": {},
    }

    prompt_hidden_states_by_key = defaultdict(list)

    for repo_id in HF_MODELS:
        for seed in SEEDS:
            for group in LANGUAGE_GROUPS:
                for language in LANGUAGE_GROUPS[group]:
                    utils.p(f"\n[green]=== {repo_id} @ {seed} @ {group} @ {language} ===[/green]")

                    config.ds.name = language

                    model = load_xpe_model(config, repo_id, seed)
                    trainer = TRAINER(model)
                    utils.p(f"Model loaded: {model}")

                    prompt_size = config.task.peft.num_virtual_tokens
                    hidden_states = first_batch_hidden_states(model, trainer)

                    # print(f"Total hidden state tensors: {len(hidden_states)}")
                    for i, hs in enumerate(hidden_states):
                        if i not in HIDDEN_STATES_LAYERS:
                            continue
                        key = get_hidden_state_key(repo_id, group, language, i)
                        # prompt_slice = hs[:, :prompt_size].detach().cpu()
                        prompt_slice = hs.detach().cpu()
                        prompt_slice = prompt_slice.reshape(-1, prompt_slice.shape[-1])
                        # randomly filter out 20% of the prompt_slice
                        prompt_slice = prompt_slice[torch.randperm(prompt_slice.shape[0])[:int(prompt_slice.shape[0] * RANDOM_PICK)]]
                        prompt_hidden_states_by_key[key].append(prompt_slice)

                    del trainer, model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if TEST_MODE: break
                if TEST_MODE: break
            if TEST_MODE: break
        if TEST_MODE: break


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

    pf["prompt_hidden_states"] = {
        key: torch.cat(tensors, dim=0)
        for key, tensors in prompt_hidden_states_by_key.items()
    }

    print("\n[bold yellow]PEFT prompt_hidden_states key structure:[/bold yellow]")
    print_dict_tree(pf)
    if TEST_MODE: exit()

    output_dir = os.path.join(MODEL.stor_path, "xlmr", "FacebookAI|xlm-roberta-large", "prompt_hidden_states")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prompt_hidden_states_5_13.pt")
    torch.save(pf, output_path)
    utils.p(f"\n[bold green]Saved prompt hidden states to {output_path}[/bold green]")
    # /home/bmikaberidze/XPE/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/prompt_hidden_states/prompt_hidden_states.pt



# - prompt_hidden_states
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@0
#     Tensor shape=(100, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@1
#     Tensor shape=(100, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@12
#     Tensor shape=(100, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@13
#     Tensor shape=(100, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@23
#     Tensor shape=(100, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@24
#     Tensor shape=(100, 1024)

# - prompt_hidden_states_5_13
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@0
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@1
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@2
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@12
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@13
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@23
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@24
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@urd_Arab@0
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@urd_Arab@1
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@urd_Arab@2
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@urd_Arab@12
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@urd_Arab@13
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@urd_Arab@23
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@urd_Arab@24
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@uig_Arab@0
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@uig_Arab@1
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@uig_Arab@2
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@uig_Arab@12
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@uig_Arab@13
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@uig_Arab@23
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@uig_Arab@24
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@azb_Arab@0
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@azb_Arab@1
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@azb_Arab@2
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@azb_Arab@12
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@azb_Arab@13
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@azb_Arab@23
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@azb_Arab@24
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@snd_Arab@0
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@snd_Arab@1
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@snd_Arab@2
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@snd_Arab@12
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@snd_Arab@13
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@snd_Arab@23
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@snd_Arab@24
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ben_Beng@0
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ben_Beng@1
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ben_Beng@2
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ben_Beng@12
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ben_Beng@13
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ben_Beng@23
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ben_Beng@24
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@0
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@1
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@2
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@12
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@13
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@23
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@24
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@0
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@1
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@2
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@12
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@13
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@23
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@24
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@eus_Latn@0
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@eus_Latn@1
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@eus_Latn@2
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@eus_Latn@12
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@eus_Latn@13
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@eus_Latn@23
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@eus_Latn@24
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@cat_Latn@0
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@cat_Latn@1
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@cat_Latn@2
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@cat_Latn@12
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@cat_Latn@13
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@cat_Latn@23
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@cat_Latn@24
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@fin_Latn@0
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@fin_Latn@1
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@fin_Latn@2
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@fin_Latn@12
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@fin_Latn@13
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@fin_Latn@23
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@fin_Latn@24
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@vie_Latn@0
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@vie_Latn@1
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@vie_Latn@2
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@vie_Latn@12
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@vie_Latn@13
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@vie_Latn@23
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@vie_Latn@24
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@mya_Mymr@0
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@mya_Mymr@1
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@mya_Mymr@2
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@mya_Mymr@12
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@mya_Mymr@13
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@mya_Mymr@23
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@mya_Mymr@24
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@knc_Arab@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@knc_Arab@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@knc_Arab@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@knc_Arab@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@knc_Arab@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@knc_Arab@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@knc_Arab@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@min_Arab@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@min_Arab@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@min_Arab@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@min_Arab@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@min_Arab@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@min_Arab@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@min_Arab@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bjn_Arab@0
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bjn_Arab@1
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bjn_Arab@2
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bjn_Arab@12
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bjn_Arab@13
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bjn_Arab@23
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bjn_Arab@24
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ace_Arab@0
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ace_Arab@1
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ace_Arab@2
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ace_Arab@12
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ace_Arab@13
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ace_Arab@23
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ace_Arab@24
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@mni_Beng@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@mni_Beng@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@mni_Beng@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@mni_Beng@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@mni_Beng@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@mni_Beng@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@mni_Beng@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@0
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@1
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@2
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@12
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@13
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@23
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@24
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@umb_Latn@0
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@umb_Latn@1
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@umb_Latn@2
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@umb_Latn@12
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@umb_Latn@13
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@umb_Latn@23
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@umb_Latn@24
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@kam_Latn@0
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@kam_Latn@1
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@kam_Latn@2
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@kam_Latn@12
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@kam_Latn@13
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@kam_Latn@23
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@kam_Latn@24
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@nus_Latn@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@nus_Latn@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@nus_Latn@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@nus_Latn@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@nus_Latn@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@nus_Latn@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@nus_Latn@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@run_Latn@0
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@run_Latn@1
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@run_Latn@2
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@run_Latn@12
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@run_Latn@13
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@run_Latn@23
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@run_Latn@24
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@0
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@1
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@2
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@12
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@13
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@23
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@24
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@shn_Mymr@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@shn_Mymr@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@shn_Mymr@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@shn_Mymr@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@shn_Mymr@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@shn_Mymr@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@shn_Mymr@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@0
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@1
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@2
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@12
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@13
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@23
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@24
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@urd_Arab@0
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@urd_Arab@1
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@urd_Arab@2
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@urd_Arab@12
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@urd_Arab@13
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@urd_Arab@23
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@urd_Arab@24
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@uig_Arab@0
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@uig_Arab@1
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@uig_Arab@2
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@uig_Arab@12
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@uig_Arab@13
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@uig_Arab@23
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@uig_Arab@24
#     Tensor shape=(4350, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@azb_Arab@0
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@azb_Arab@1
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@azb_Arab@2
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@azb_Arab@12
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@azb_Arab@13
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@azb_Arab@23
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@azb_Arab@24
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@snd_Arab@0
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@snd_Arab@1
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@snd_Arab@2
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@snd_Arab@12
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@snd_Arab@13
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@snd_Arab@23
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@snd_Arab@24
#     Tensor shape=(3960, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ben_Beng@0
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ben_Beng@1
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ben_Beng@2
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ben_Beng@12
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ben_Beng@13
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ben_Beng@23
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ben_Beng@24
#     Tensor shape=(4480, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@0
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@1
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@2
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@12
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@13
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@23
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@24
#     Tensor shape=(4090, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@0
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@1
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@2
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@12
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@13
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@23
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@24
#     Tensor shape=(3320, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@eus_Latn@0
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@eus_Latn@1
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@eus_Latn@2
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@eus_Latn@12
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@eus_Latn@13
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@eus_Latn@23
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@eus_Latn@24
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@cat_Latn@0
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@cat_Latn@1
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@cat_Latn@2
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@cat_Latn@12
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@cat_Latn@13
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@cat_Latn@23
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@cat_Latn@24
#     Tensor shape=(4220, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@fin_Latn@0
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@fin_Latn@1
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@fin_Latn@2
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@fin_Latn@12
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@fin_Latn@13
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@fin_Latn@23
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@fin_Latn@24
#     Tensor shape=(3710, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@vie_Latn@0
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@vie_Latn@1
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@vie_Latn@2
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@vie_Latn@12
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@vie_Latn@13
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@vie_Latn@23
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@vie_Latn@24
#     Tensor shape=(3580, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@mya_Mymr@0
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@mya_Mymr@1
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@mya_Mymr@2
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@mya_Mymr@12
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@mya_Mymr@13
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@mya_Mymr@23
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@mya_Mymr@24
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@knc_Arab@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@knc_Arab@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@knc_Arab@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@knc_Arab@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@knc_Arab@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@knc_Arab@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@knc_Arab@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@min_Arab@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@min_Arab@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@min_Arab@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@min_Arab@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@min_Arab@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@min_Arab@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@min_Arab@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bjn_Arab@0
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bjn_Arab@1
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bjn_Arab@2
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bjn_Arab@12
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bjn_Arab@13
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bjn_Arab@23
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bjn_Arab@24
#     Tensor shape=(5240, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ace_Arab@0
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ace_Arab@1
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ace_Arab@2
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ace_Arab@12
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ace_Arab@13
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ace_Arab@23
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ace_Arab@24
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@mni_Beng@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@mni_Beng@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@mni_Beng@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@mni_Beng@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@mni_Beng@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@mni_Beng@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@mni_Beng@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@0
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@1
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@2
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@12
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@13
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@23
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@24
#     Tensor shape=(5630, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@umb_Latn@0
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@umb_Latn@1
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@umb_Latn@2
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@umb_Latn@12
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@umb_Latn@13
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@umb_Latn@23
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@umb_Latn@24
#     Tensor shape=(4600, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@kam_Latn@0
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@kam_Latn@1
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@kam_Latn@2
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@kam_Latn@12
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@kam_Latn@13
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@kam_Latn@23
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@kam_Latn@24
#     Tensor shape=(4990, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@nus_Latn@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@nus_Latn@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@nus_Latn@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@nus_Latn@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@nus_Latn@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@nus_Latn@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@nus_Latn@24
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@run_Latn@0
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@run_Latn@1
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@run_Latn@2
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@run_Latn@12
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@run_Latn@13
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@run_Latn@23
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@run_Latn@24
#     Tensor shape=(5370, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@0
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@1
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@2
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@12
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@13
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@23
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@24
#     Tensor shape=(4860, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@shn_Mymr@0
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@shn_Mymr@1
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@shn_Mymr@2
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@shn_Mymr@12
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@shn_Mymr@13
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@shn_Mymr@23
#     Tensor shape=(5880, 1024)
#   - mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@shn_Mymr@24
#     Tensor shape=(5880, 1024)
