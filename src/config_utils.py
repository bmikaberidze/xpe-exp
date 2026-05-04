from micm_nlp import utils

import os
import random
import pandas as pd
from peft import PeftType
from types import SimpleNamespace
from micm_nlp.models.model import MODEL
from micm_nlp.config import CONFIG
from micm_nlp.enums import ConfigTypeSE, ModelArchSE
from micm_nlp.models.xpe import CrossPromptEncoderReparameterizationType as XPE_RE

ds_path = "artefacts/datasets"
def get_sib200_meta():
    meta_path = f'{ds_path}/benchmarks/text_classification/topic/sib200_tokenized_xlmr/meta.csv'
    meta = pd.read_csv(meta_path)
    return meta

sib200_meta = get_sib200_meta()
sib200_ds_names = sib200_meta['code'].tolist()
xlmr_seen_sib200_ds_names = sib200_meta[sib200_meta['xlmr'] == 1]['code'].tolist()
xlmr_unseen_sib200_ds_names = list(set(sib200_ds_names) - set(xlmr_seen_sib200_ds_names))

xlmr_unseen_script_sib200_ds_names = ['nqo_Nkoo', 'sat_Olck', 'taq_Tfng', 'tzm_Tfng', 'bod_Tibt', 'dzo_Tibt']

# arb_Arab, deu_Latn, eng_Latn, fra_Latn, spa_Latn, jpn_Jpan, zho_Hans
joshi5_sib200_ds_names = sib200_meta[sib200_meta['class'] == 5]['code'].tolist()
not_joshi5_sib200_ds_names = list(set(sib200_ds_names) - set(joshi5_sib200_ds_names))

Sib200_family_to_id = {
    'Afro-Asiatic': 0,
    'Atlantic-Congo': 1,
    'Austroasiatic': 2,
    'Austronesian': 3,
    'Aymaran': 4,
    'Basque': 5,
    'Constructed': 6,
    'Dravidian': 7,
    'Indo-European': 8,
    'Japonic': 9,
    'Kartvelian': 10,
    'Koreanic': 11,
    'Mande': 12,
    'Mongolic-Khitan': 13,
    'Nilotic': 14,
    'Quechuan': 15,
    'Sino-Tibetan': 16,
    'Tai-Kadai': 17,
    'Tupian': 18,
    'Turkic': 19,
    'Uralic': 20
}

def get_sib200_LID_LFID(lang_code):
    lang_row = sib200_meta[sib200_meta['code'] == lang_code].iloc[0]
    LFID = Sib200_family_to_id[lang_row['family']]
    LID = int(lang_row.name)
    return LID, LFID

# print(get_sib200_LID_LFID('afr_Latn'))
# exit()

# xlmr_seen_lrl_sib200_ds_names  = sib200_meta[(sib200_meta['class'].isin([0, 1, 2])) & (sib200_meta['xlmr'] == 1)]['code'].tolist()
# xlmr_unseen_lrl_sib200_ds_names = sib200_meta[(sib200_meta['class'].isin([0, 1, 2])) & (sib200_meta['xlmr'] == 0)]['code'].tolist()
# random_seen = random.sample(xlmr_seen_lrl_sib200_ds_names, min(25, len(xlmr_seen_lrl_sib200_ds_names)))
# random_unseen = random.sample(xlmr_unseen_lrl_sib200_ds_names, min(25, len(xlmr_unseen_lrl_sib200_ds_names)))
# print("Random 25 from seen:", random_seen)
# print("Random 25 from unseen:", random_unseen)
xlmr_25_seen_lrl_sib200_ds_names = ['asm_Beng', 'swh_Latn', 'plt_Latn', 'ory_Orya', 'guj_Gujr', 'khm_Khmr', 'zho_Hant', 'uig_Arab', 'nob_Latn', 'xho_Latn', 'sun_Latn', 'jav_Latn', 'mal_Mlym', 'pan_Guru', 'amh_Ethi', 'mar_Deva', 'cym_Latn', 'som_Latn', 'ydd_Hebr', 'pbt_Arab', 'gle_Latn', 'snd_Arab', 'gaz_Latn', 'nno_Latn', 'hye_Armn']
xlmr_25_unseen_lrl_sib200_ds_names = ['szl_Latn', 'yor_Latn', 'ssw_Latn', 'ace_Latn', 'lmo_Latn', 'kas_Arab', 'pag_Latn', 'min_Arab', 'tuk_Latn', 'ltg_Latn', 'run_Latn', 'lin_Latn', 'fij_Latn', 'tum_Latn', 'ckb_Arab', 'tat_Cyrl', 'kac_Latn', 'tsn_Latn', 'ltz_Latn', 'kik_Latn', 'lim_Latn', 'crh_Latn', 'twi_Latn', 'scn_Latn', 'sot_Latn']

xlmr_24_divers_seen_sib200_ds_names = [
    "tir_Ethi", "arz_Arab", #"kab_Latn", issue
    "swh_Latn", "xho_Latn",
    "khm_Khmr", "vie_Latn",
    "ind_Latn", "zsm_Latn",
    "eus_Latn",
    "epo_Latn",
    "hye_Armn", "bos_Latn",
    "kat_Geor",
    "kor_Hang",
    "khk_Cyrl",
    "mya_Mymr", "zho_Hant",
    "lao_Laoo", "tha_Thai",
    "uig_Arab", "kaz_Cyrl",
    "fin_Latn", "hun_Latn"
]

xlmr_24_divers_unseen_sib200_ds_names = [
    "ajp_Arab", "amh_Ethi", "kab_Latn"
    "yor_Latn", "sna_Latn", "run_Latn",
    "ceb_Latn", "bjn_Arab", "ban_Latn",
    "ayr_Latn",
    "ckb_Arab", "mag_Deva", "sin_Sinh",
    "dyu_Latn",
    "knc_Arab",
    "quy_Latn",
    "lus_Latn", "mni_Beng", "yue_Hant",
    "shn_Mymr",
    "grn_Latn",
    "bak_Cyrl", "crh_Latn", "tat_Cyrl"
]

xlmr_seen_family_groups_sib200_ds_names = [
    # Afro-Asiatic
    'hau_Latn',
    'heb_Hebr',
    'arb_Arab',
    # Atlantic-Congo
    'swh_Latn',
    'xho_Latn',
    # Austroasiatic
    'khm_Khmr',
    'vie_Latn',
    # Austronesian
    'sun_Latn',
    'ind_Latn',
    'zsm_Latn',
    # Indo-European
    'eng_Latn',
    'fra_Latn',
    'spa_Latn',
    # Sino-Tibetan
    'mya_Mymr',
    'zho_Hant',
    'zho_Hans',
    # Tai-Kadai
    'lao_Laoo',
    'tha_Thai',
    # Turkic
    'kaz_Cyrl',
    'uzn_Latn',
    'tur_Latn'
]

enarzho_sib200_ds_names = [ 'eng_Latn', 'arb_Arab', 'zho_Hans' ]

source_ds_name = 'source'
def get_ds_names(benchmark_name, llm):

    # EN ================================================
    if benchmark_name == 'sib200_en':
        return 'eng_Latn', ['eng_Latn'], list(set(xlmr_seen_sib200_ds_names) - set(['eng_Latn'])) + xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_en_seen':
        return 'eng_Latn', ['eng_Latn'], list(set(xlmr_seen_sib200_ds_names) - set(['eng_Latn']))
        
    if benchmark_name == 'sib200_en_unseen':
        return 'eng_Latn', ['eng_Latn'], xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_en_ablation':
        return 'eng_Latn', ['eng_Latn'], xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names

    # ENARZHO ================================================
    if benchmark_name == 'sib200_enarzho':
        return f'{source_ds_name}_{llm}_enarzho', enarzho_sib200_ds_names, not_joshi5_sib200_ds_names
    
    if benchmark_name == 'sib200_enarzho_ablation':
        return f'{source_ds_name}_{llm}_enarzho', enarzho_sib200_ds_names, xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names

    if benchmark_name == 'sib200_enarzho_divers_ablation':
        return f'{source_ds_name}_{llm}_enarzho', enarzho_sib200_ds_names, xlmr_24_divers_seen_sib200_ds_names + xlmr_24_divers_unseen_sib200_ds_names
    
    # Joshi5 ================================================
    if benchmark_name == 'sib200_joshi5':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, not_joshi5_sib200_ds_names
    
    if benchmark_name == 'sib200_joshi5_ablation':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names
    
    if benchmark_name in [
        'sib200_joshi5_divers_ablation', 
        'sib200_joshi5_divers_24'
    ]:
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_24_divers_seen_sib200_ds_names + xlmr_24_divers_unseen_sib200_ds_names
    
    if benchmark_name == 'sib200_joshi5_divers_ablation_unseen':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_24_divers_unseen_sib200_ds_names

    if benchmark_name == 'sib200_joshi5_abl_unseen':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_25_unseen_lrl_sib200_ds_names

    if benchmark_name == 'sib200_joshi5_unseen':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, xlmr_unseen_sib200_ds_names
    
    if benchmark_name == 'sib200_joshi5_joshi5':
        return f'{source_ds_name}_{llm}_joshi5', joshi5_sib200_ds_names, joshi5_sib200_ds_names

    # XLM-R seen ================================================
    if benchmark_name == 'sib200_xlmr_seen':
        return f'{source_ds_name}_{llm}_seen', xlmr_seen_sib200_ds_names, xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_xlmr_seen_family_groups':
        return f'{source_ds_name}_{llm}_seen_family_groups', xlmr_seen_family_groups_sib200_ds_names, xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_xlmr_seen_seen':
        return f'{source_ds_name}_{llm}_seen', xlmr_seen_sib200_ds_names, xlmr_seen_sib200_ds_names
    
    if benchmark_name == 'sib200_xlmr_seen_abl_unseen':
        return f'{source_ds_name}_{llm}_seen', xlmr_seen_sib200_ds_names, xlmr_25_unseen_lrl_sib200_ds_names
    
    if benchmark_name == 'sib200_xlmr_seen_ablation':
        return f'{source_ds_name}_{llm}_seen', xlmr_seen_sib200_ds_names, xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names

    # XLM-R unseen ================================================
    if benchmark_name == 'sib200_xlmr_unseen':        
        return f'{source_ds_name}_{llm}_unseen', xlmr_unseen_sib200_ds_names, xlmr_seen_sib200_ds_names

    # All ================================================
    if benchmark_name == 'sib200_all_xlmr_unseen':
        return f'{source_ds_name}_{llm}_all', sib200_ds_names, xlmr_unseen_sib200_ds_names

    if benchmark_name == 'sib200_all_ablation':
        return f'{source_ds_name}_{llm}_all', sib200_ds_names, xlmr_25_seen_lrl_sib200_ds_names + xlmr_25_unseen_lrl_sib200_ds_names

    if benchmark_name == 'sib200_all':
        return f'{source_ds_name}_{llm}_all', sib200_ds_names, sib200_ds_names
    
    raise ValueError(f'Invalid benchmark_name: {benchmark_name}')

# 
# Get Configuration by SLURM Task
# 

def get_config_by_slurm_task(config_name, ds_name, supervision_regime):
    source_model_uuid4 = None

    s_job_id = int(os.getenv('SLURM_JOB_ID', 0))
    s_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))

    config = CONFIG.load(config_name, ConfigTypeSE.LANGUAGE_MODEL)
    s_task_conf_id = s_task_id

    s_task_configs = {
        # test run
        0: (f'TEST', 20, 0, 2, None, 10, 32, 0.00005, 'adafactor', 24, XPE_RE.NONE, '', 0.005, 0.0, False),

        # XLT run on enarzho, joshi5, and seen
        1: (f'SPT', 20, 0, 2, None, 10, 32, 0.00005, 'adafactor', 24000, XPE_RE.NONE, '', 0.005, 0.0, False),
        2: (f'D30', 20, 0.3, 2, None, 10, 32, 0.00005, 'adafactor', 24000, XPE_RE.MLP, '', 0.005, 0.0, False),
        3: (f'D70', 20, 0.7, 2, None, 10, 32, 0.00005, 'adafactor', 24000, XPE_RE.MLP, '', 0.005, 0.0, False),
        4: (f'XPE', 20, 1, 2, None, 10, 32, 0.00005, 'adafactor', 24000, XPE_RE.MLP, '', 0.005, 0.0, False),
        
        # LID over joshi5, and seen.
        101: (f'SPT_SEEN', 20, 0, 2, None, 5, 32, 0.00005, 'adafactor', 0, XPE_RE.NONE, '', 0, 0.0, True),
        102: (f'XPE_SEEN', 20, 1, 2, None, 5, 32, 0.00005, 'adafactor', 0, XPE_RE.MLP, '', 0, 0.0, True),
        103: (f'SPT_J5', 20, 0, 2, None, 5, 32, 0.00005, 'adafactor', 0, XPE_RE.NONE, '', 0, 0.0, True),
        104: (f'XPE_J5', 20, 1, 2, None, 5, 32, 0.00005, 'adafactor', 0, XPE_RE.MLP, '', 0, 0.0, True),
    }

    conf = s_task_configs[s_task_conf_id]

    if supervision_regime:
        s_task_conf_name = f'F_{conf[0]}'
        config.test.zero_shot_only = False
    else:
        s_task_conf_name = f'Z_{conf[0]}'
        config.test.zero_shot = True
        config.test.zero_shot_only = True

    config.task.peft.num_virtual_tokens = conf[1]
    config.task.peft.encoder_ratio = conf[2]
    config.task.peft.encoder_num_layers = conf[3]
    config.task.peft.encoder_input_size = conf[4]
    config.training_args.num_train_epochs = conf[5]
    config.training_args.per_device_train_batch_size = conf[6]
    config.training_args.learning_rate = conf[7]
    lr_type = conf[8]
    if lr_type == 'adafactor':
        config.training_args.optim = lr_type
        config.training_args.optim_args = SimpleNamespace(
            beta1=None,
            decay_rate=-0.8,
            weight_decay=0.01,
            clip_threshold=1.0,
            scale_parameter=True,
            relative_step=True,
            warmup_init=True
        )

    max_steps = conf[9]
    if max_steps: config.training_args.max_steps = max_steps

    encoder_type = conf[10]
    config.task.peft.encoder_reparameterization_type = encoder_type

    if len(conf) > 11:
        source_model_uuid4 = conf[11]
    if len(conf) > 12:
        spec_lr = conf[12]
        if spec_lr:
            config.custom_training_args.optimizer_grouped_parameters[0].lr = spec_lr
        spec_wd = conf[13]
        config.custom_training_args.optimizer_grouped_parameters[0].weight_decay = spec_wd
        if encoder_type == XPE_RE.NONE or config.model.architecture == ModelArchSE.XGLM:
            config.custom_training_args.optimizer_grouped_parameters[0].param_name_parts = ['embedding'] # concerns all prompt encoder embeddings
        elif encoder_type == XPE_RE.MLP:
            config.custom_training_args.optimizer_grouped_parameters[0].param_name_parts = ['xpe_embedding'] # concerns only dedicated embeddings
    if len(conf) > 14:
        config.task.peft.encoder_freeze = conf[14]

    return config, s_task_conf_name, f'{s_job_id}_{s_task_id}', source_model_uuid4

# 

ds_base_dirs = ''
def set_ds_dirs(config, name, suffix_dirs = None, subset_dirs = False):
    global ds_base_dirs
    if not ds_base_dirs: ds_base_dirs = config.ds.dirs
    config.ds.descriptive_name = name
    config.ds.dirs = f'{ds_base_dirs}/{name}' if name else ds_base_dirs
    config.ds.dirs = f'{config.ds.dirs}/{suffix_dirs}' if suffix_dirs else config.ds.dirs
    config.ds.dirs = f'{config.ds.dirs}/subset/{subset_dirs}' if subset_dirs else config.ds.dirs
    return config

encoder_freeze = None
num_train_epochs = 0
max_steps = 0
zero_shot_only = None
zero_shot = None
batch_size = 0
def update_config(config, ds_name, task_id, suffix_dirs = None, 
                  preproc_subset_ratio = None, subset_dirs = False, 
                  source_model_uuid4 = None, source_training = False):
    
    config = set_ds_dirs(config, ds_name, suffix_dirs, subset_dirs)
    if preproc_subset_ratio:
        config.ds.preproc_rules.subset.use = preproc_subset_ratio

    # Tasks
    config.task.id = task_id
            
    # Target Preparations
    if getattr(config.task, 'peft', None):

        global num_train_epochs, max_steps, zero_shot, zero_shot_only, encoder_freeze, batch_size
        batch_size = batch_size or config.training_args.per_device_train_batch_size
        zero_shot = zero_shot or config.test.zero_shot
        zero_shot_only = zero_shot_only if zero_shot_only is not None else config.test.zero_shot_only
        encoder_freeze = encoder_freeze if encoder_freeze is not None else config.task.peft.encoder_freeze
        num_train_epochs = num_train_epochs or config.training_args.num_train_epochs
        max_steps = max_steps or getattr(config.training_args, 'max_steps', None)

        if not source_training:
            utils.p(f'\n[bold green]Target Training Phase![/bold green]')

            assert source_model_uuid4, 'source_model_uuid4 is required for target training'
            utils.p(f'\nsource_model_uuid4: {source_model_uuid4}')

            config.test.zero_shot = zero_shot
            config.test.zero_shot_only = zero_shot_only
            config.task.peft.encoder_freeze = encoder_freeze
            config.custom_training_args.early_stopping_patience = 30
            config.training_args.per_device_train_batch_size = batch_size

            config.training_args.warmup_ratio = 0.05
            config.training_args.weight_decay = 0.01
            config.training_args.max_grad_norm = 1.0

            if max_steps:
                config.training_args.max_steps = int(max_steps/4)
            else:
                config.training_args.num_train_epochs = num_train_epochs * 5

            if config.task.peft.peft_type == PeftType.P_TUNING:
                config.task.peft.encoder_dropout = 0.0
                config.task.peft.encoder_init_state_dict_path = MODEL.get_last_checkpoint_path_by_uuid4(source_model_uuid4)
        
        elif source_training:
            
            utils.p(f'\n[bold green]Source Training Phase![/bold green]')

            config.test.zero_shot = False
            config.test.zero_shot_only = False
            config.task.peft.encoder_freeze = False
            config.custom_training_args.early_stopping_patience = 20
            config.training_args.per_device_train_batch_size = 32

            config.training_args.warmup_ratio = 0.1
            config.training_args.weight_decay = 0.01
            config.training_args.max_grad_norm = 1.0

            if max_steps:
                config.training_args.max_steps = max_steps
            else:
                config.training_args.num_train_epochs = num_train_epochs

            if config.task.peft.peft_type == PeftType.P_TUNING:
                config.task.peft.encoder_dropout = 0.1
                config.task.peft.encoder_init_state_dict_path = None

    return config

# ================================
# Trained Model Paths
# ================================
trained_model_paths = {
    # Source models for LID
    # "spt_seen":
    "70c56fb4-f6e5-40b5-bfbe-5fdd0587cc7d": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/70c56fb4-f6e5-40b5-bfbe-5fdd0587cc7d_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "932b3f65-0ca4-4e07-9abe-a0e2fbfbaa6f": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/932b3f65-0ca4-4e07-9abe-a0e2fbfbaa6f_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "78520010-9059-4daa-8b6a-86de2fffd6c6": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/78520010-9059-4daa-8b6a-86de2fffd6c6_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "ad872348-a9fb-46ac-a59d-722fb6f0262b": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/ad872348-a9fb-46ac-a59d-722fb6f0262b_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "5bc5659f-c51c-4b67-9c6c-78416fcb8d4f": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/5bc5659f-c51c-4b67-9c6c-78416fcb8d4f_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "4de99566-48da-4f22-8273-619f6d601361": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/4de99566-48da-4f22-8273-619f6d601361_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "7bed6746-baae-4716-a14b-2073fd9a13b5": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/7bed6746-baae-4716-a14b-2073fd9a13b5_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "d5352222-c549-4861-b4e9-353fc59eed2f": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/d5352222-c549-4861-b4e9-353fc59eed2f_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "8f405d0e-d752-4008-b710-e90bb1248ee2": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/8f405d0e-d752-4008-b710-e90bb1248ee2_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "b078674c-086e-4326-8354-139638092aaa": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/b078674c-086e-4326-8354-139638092aaa_FacebookAI|xlm-roberta-large_2020333_4_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    # "xpe_seen":
    "10e025f6-1da6-44a4-9f96-b8893e65c2c5": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/10e025f6-1da6-44a4-9f96-b8893e65c2c5_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "6ff52a90-aaa0-4d1a-b0b2-926bd57cd5c5": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/6ff52a90-aaa0-4d1a-b0b2-926bd57cd5c5_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "aef2b130-debc-499e-aec3-f14d6c5fbb1a": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/aef2b130-debc-499e-aec3-f14d6c5fbb1a_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "e8a21d23-789e-4cbf-82e4-27072ded5f3e": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/e8a21d23-789e-4cbf-82e4-27072ded5f3e_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "40d575f5-05d8-45a0-b599-36a593e37959": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/40d575f5-05d8-45a0-b599-36a593e37959_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "e10d1637-0452-4e4b-a339-37e5d6134fec": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/e10d1637-0452-4e4b-a339-37e5d6134fec_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "a6096917-2d98-47ed-92e3-ba00ef530f48": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/a6096917-2d98-47ed-92e3-ba00ef530f48_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "03a9d451-3312-4748-810d-eedc8a4009be": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/03a9d451-3312-4748-810d-eedc8a4009be_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "ad56d282-e924-4026-b8fb-691721266098": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/ad56d282-e924-4026-b8fb-691721266098_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    "ea1f72c6-0721-487c-bfa6-114a6cb4d6ce": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/ea1f72c6-0721-487c-bfa6-114a6cb4d6ce_FacebookAI|xlm-roberta-large_2021036_1_10_32_text_classification|topic|sib200_hf|source_xlmr_seen|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_seen",
    # "spt_joshi5":
    "c417a084-008a-4cbc-86b8-d703a2aad405": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/c417a084-008a-4cbc-86b8-d703a2aad405_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "aab0f095-9be9-493e-8cb4-d2e0f671f7fd": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/aab0f095-9be9-493e-8cb4-d2e0f671f7fd_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "29405be4-0b55-4982-8acc-759bed8adbb9": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/29405be4-0b55-4982-8acc-759bed8adbb9_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "bf5be0bb-ed5e-4d13-a095-2b3a1dbd5562": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/bf5be0bb-ed5e-4d13-a095-2b3a1dbd5562_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "e3ae555d-7516-4ec7-92ee-65e7c8a8da61": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/e3ae555d-7516-4ec7-92ee-65e7c8a8da61_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "270a6eda-5c6f-4dd2-8d65-8050c07e6e92": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/270a6eda-5c6f-4dd2-8d65-8050c07e6e92_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "cffc931d-59e1-434b-86c4-29a0b45b58f1": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/cffc931d-59e1-434b-86c4-29a0b45b58f1_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "cb9acdd3-fbe1-4ca8-aeec-c5c1d7dab2e0": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/cb9acdd3-fbe1-4ca8-aeec-c5c1d7dab2e0_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "dc1c7cfa-ef0a-4ad1-88f4-d7f8df06bdbf": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/dc1c7cfa-ef0a-4ad1-88f4-d7f8df06bdbf_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "a749d626-a7f5-40dd-a4e1-678bf7868c67": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/a749d626-a7f5-40dd-a4e1-678bf7868c67_FacebookAI|xlm-roberta-large_2020335_4_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    # "xpe_joshi5":
    "ed723b64-47e9-4ad7-ab61-f75c3f2cfd2b": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/ed723b64-47e9-4ad7-ab61-f75c3f2cfd2b_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "a3b67bca-6382-45dd-a991-2467c27483b1": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/a3b67bca-6382-45dd-a991-2467c27483b1_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "4a72123f-17c4-4565-a9ff-a76ada3eb6a1": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/4a72123f-17c4-4565-a9ff-a76ada3eb6a1_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "922c5a8f-f0b4-40c6-abab-a60f785151be": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/922c5a8f-f0b4-40c6-abab-a60f785151be_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "932bb7f9-ee32-421c-90c0-fad65ce26038": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/932bb7f9-ee32-421c-90c0-fad65ce26038_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "387097bd-6c91-42d1-8565-72a03cbd0703": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/387097bd-6c91-42d1-8565-72a03cbd0703_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "61a28706-ef12-46ec-b0b8-431199c272b9": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/61a28706-ef12-46ec-b0b8-431199c272b9_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "3253362d-a2b8-42a1-b1cb-ef28dd199324": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/3253362d-a2b8-42a1-b1cb-ef28dd199324_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "6ff741ed-ad58-4ac9-9bb6-03b0e46a9535": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/6ff741ed-ad58-4ac9-9bb6-03b0e46a9535_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
    "7b0367b9-888f-4f17-96cd-463010d1e36a": "/fscratch/bmikaberidze/group5_nlp/nlpka/models/storage/xlmr/FacebookAI|xlm-roberta-large/topic/7b0367b9-888f-4f17-96cd-463010d1e36a_FacebookAI|xlm-roberta-large_2021075_1_10_32_text_classification|topic|sib200_hf|source_xlmr_joshi5|tokenized|FacebookAI|xlm-roberta-large_source_xlmr_joshi5",
}

# ================================
# Grouped Trained Models
# ================================
grouped_trained_models = {
    "SPT_SEEN": [
        "70c56fb4-f6e5-40b5-bfbe-5fdd0587cc7d",
        "932b3f65-0ca4-4e07-9abe-a0e2fbfbaa6f",
        "78520010-9059-4daa-8b6a-86de2fffd6c6",
        "ad872348-a9fb-46ac-a59d-722fb6f0262b",
        "5bc5659f-c51c-4b67-9c6c-78416fcb8d4f",
        "4de99566-48da-4f22-8273-619f6d601361",
        "7bed6746-baae-4716-a14b-2073fd9a13b5",
        "d5352222-c549-4861-b4e9-353fc59eed2f",
        "8f405d0e-d752-4008-b710-e90bb1248ee2",
        "b078674c-086e-4326-8354-139638092aaa"
    ],
    "XPE_SEEN": [
        "10e025f6-1da6-44a4-9f96-b8893e65c2c5",
        "6ff52a90-aaa0-4d1a-b0b2-926bd57cd5c5",
        "aef2b130-debc-499e-aec3-f14d6c5fbb1a",
        "e8a21d23-789e-4cbf-82e4-27072ded5f3e",
        "40d575f5-05d8-45a0-b599-36a593e37959",
        "e10d1637-0452-4e4b-a339-37e5d6134fec",
        "a6096917-2d98-47ed-92e3-ba00ef530f48",
        "03a9d451-3312-4748-810d-eedc8a4009be",
        "ad56d282-e924-4026-b8fb-691721266098",
        "ea1f72c6-0721-487c-bfa6-114a6cb4d6ce"
    ],
    "SPT_J5": [
        "c417a084-008a-4cbc-86b8-d703a2aad405",
        "aab0f095-9be9-493e-8cb4-d2e0f671f7fd",
        "29405be4-0b55-4982-8acc-759bed8adbb9",
        "bf5be0bb-ed5e-4d13-a095-2b3a1dbd5562",
        "e3ae555d-7516-4ec7-92ee-65e7c8a8da61",
        "270a6eda-5c6f-4dd2-8d65-8050c07e6e92",
        "cffc931d-59e1-434b-86c4-29a0b45b58f1",
        "cb9acdd3-fbe1-4ca8-aeec-c5c1d7dab2e0",
        "dc1c7cfa-ef0a-4ad1-88f4-d7f8df06bdbf",
        "a749d626-a7f5-40dd-a4e1-678bf7868c67"
    ],
    "XPE_J5": [
        "ed723b64-47e9-4ad7-ab61-f75c3f2cfd2b",
        "a3b67bca-6382-45dd-a991-2467c27483b1",
        "4a72123f-17c4-4565-a9ff-a76ada3eb6a1",
        "922c5a8f-f0b4-40c6-abab-a60f785151be",
        "932bb7f9-ee32-421c-90c0-fad65ce26038",
        "387097bd-6c91-42d1-8565-72a03cbd0703",
        "61a28706-ef12-46ec-b0b8-431199c272b9",
        "3253362d-a2b8-42a1-b1cb-ef28dd199324",
        "6ff741ed-ad58-4ac9-9bb6-03b0e46a9535",
        "7b0367b9-888f-4f17-96cd-463010d1e36a"
    ]
}
