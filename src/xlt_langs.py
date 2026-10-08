"""Named language groups for cross-lingual transfer -- the single source of truth.

``LANG_GROUPS`` are SOURCE groups: a group config entry names one as
``source_group:`` and ``scripts/xlt_runner.py`` tunes on those languages
concatenated. ``LOW_PERF_LANG_GROUPS`` are TARGET groups, read only by the result
aggregators (``scripts/evals/``). Moved here from the pre-0.4 ``scripts/run_xlt.py``
CLI (2026-09-30); ``tests/test_lang_groups.py`` pins them.
"""

LANG_GROUPS = {

    # EnArZho (paper sec. 4): a compact, high-resource, typologically diverse set
    # of English, Arabic, Mandarin Chinese. The smallest source group (subset of
    # joshi5 / seen). Paper reduces training langs to 3 to probe the SPT/XPE mix.
    'enarzho': ['eng_Latn', 'arb_Arab', 'zho_Hans'],

    # Joshi et al. (2020) Class-5 ("The Winners"): the 7 highest-resource
    # languages. Note: German + Japanese are NOT in BLOOM's ROOTS pretraining,
    # so as BLOOM source langs they are unseen-by-pretraining (paper used XLM-R).
    'joshi5': ['eng_Latn', 'spa_Latn', 'deu_Latn', 'fra_Latn', 'jpn_Jpan', 'zho_Hans', 'arb_Arab'],

    # Aya Expanse (CohereLabs/aya-expanse-8b) 23 languages, intersected with
    # Belebele's 122 (24 codes; all 23 present, Chinese as both scripts). Native
    # scripts; MSA arb_Arab; Western Persian pes_Arab. The Aya analogue of the
    # paper's "Seen" source group (for the aya backbone, not BLOOMZ).
    'aya_seen': [
        'eng_Latn', 'fra_Latn', 'spa_Latn', 'por_Latn', 'ita_Latn', 'deu_Latn',
        'nld_Latn', 'ces_Latn', 'pol_Latn', 'ron_Latn', 'ell_Grek', 'rus_Cyrl',
        'ukr_Cyrl', 'tur_Latn', 'arb_Arab', 'heb_Hebr', 'pes_Arab', 'hin_Deva',
        'ind_Latn', 'vie_Latn', 'jpn_Jpan', 'kor_Hang', 'zho_Hans', 'zho_Hant',
    ],

    # Aya high-performing: the top 50% (61 of 122) Belebele langs by Aya's own
    # zero-shot accuracy (group 25, run 20260924_035112_probe_aya_zs, full 900
    # items, micm-nlp 0.5 era). The performance-based replacement for "Seen",
    # which needs a published pretraining list (Gemma has none). Clean cut: #61
    # ben_Beng 0.423, #62 uzn_Latn 0.417. Contains all of aya_seen and joshi5;
    # disjoint from LOW_PERF_LANG_GROUPS['aya']. Order = zero-shot rank.
    'aya_high': [
        'eng_Latn', 'fra_Latn', 'zho_Hans', 'zho_Hant', 'deu_Latn', 'ita_Latn',
        'ces_Latn', 'por_Latn', 'ron_Latn', 'ell_Grek', 'spa_Latn', 'rus_Cyrl',
        'ukr_Cyrl', 'ind_Latn', 'vie_Latn', 'pol_Latn', 'arb_Arab', 'jpn_Jpan',
        'nld_Latn', 'kor_Hang', 'heb_Hebr', 'pes_Arab', 'tur_Latn', 'slk_Latn',
        'swe_Latn', 'zsm_Latn', 'cat_Latn', 'dan_Latn', 'afr_Latn', 'nob_Latn',
        'hin_Deva', 'bul_Cyrl', 'hrv_Latn', 'arz_Arab', 'lit_Latn', 'apc_Arab',
        'ars_Arab', 'acm_Arab', 'srp_Cyrl', 'slv_Latn', 'mkd_Cyrl', 'hun_Latn',
        'fin_Latn', 'ary_Arab', 'tgl_Latn', 'azj_Latn', 'urd_Arab', 'hin_Latn',
        'tha_Thai', 'tam_Taml', 'lvs_Latn', 'isl_Latn', 'jav_Latn', 'ckb_Arab',
        'mar_Deva', 'mal_Mlym', 'kea_Latn', 'kat_Geor', 'est_Latn', 'als_Latn',
        'ben_Beng',
    ],

    # Gemma 4 high-performing: the top 50% (61 of 122) Belebele langs by
    # gemma-4-12B-it's own zero-shot accuracy through its chat template (group 25,
    # run 20261008_002553_g4_12b_it_prefill, full 900 items, micm-nlp 0.5 era).
    # Same rule as aya_high. The cut is NOT clean: #61 kat_Geor 0.8311, #62
    # hin_Deva 0.8300 -- one item in 900 apart -- so hin_Deva's exclusion is the
    # rule, not a gap in the data. Contains joshi5 and enarzho. Order = zero-shot rank.
    'gemma_high': [
        'eng_Latn', 'fra_Latn', 'por_Latn', 'spa_Latn', 'deu_Latn', 'ron_Latn',
        'bul_Cyrl', 'rus_Cyrl', 'arb_Arab', 'pol_Latn', 'nob_Latn', 'dan_Latn',
        'cat_Latn', 'ita_Latn', 'zho_Hans', 'vie_Latn', 'swe_Latn', 'nld_Latn',
        'ukr_Cyrl', 'ind_Latn', 'ell_Grek', 'zsm_Latn', 'zho_Hant', 'fin_Latn',
        'slk_Latn', 'hrv_Latn', 'afr_Latn', 'slv_Latn', 'tgl_Latn', 'kor_Hang',
        'mkd_Cyrl', 'hun_Latn', 'ces_Latn', 'srp_Cyrl', 'heb_Hebr', 'tur_Latn',
        'lvs_Latn', 'pes_Arab', 'isl_Latn', 'jpn_Jpan', 'als_Latn', 'swh_Latn',
        'lit_Latn', 'uzn_Latn', 'tha_Thai', 'est_Latn', 'jav_Latn', 'ars_Arab',
        'arz_Arab', 'pan_Guru', 'mar_Deva', 'hye_Armn', 'urd_Arab', 'ceb_Latn',
        'ben_Beng', 'ory_Orya', 'guj_Gujr', 'azj_Latn', 'apc_Arab', 'sin_Sinh',
        'kat_Geor',
    ],

    # Grid 8 (BLOOMZ Belebele self-split, 2026-06-09): the 7 languages the old CLI took
    # as --source-langs, before named groups existed. Kept so that grid's rewritten
    # group config (config/groups/8_bebe_self_split_bloomz.self_split7.yml) can name it.
    'self_split7': ['arb_Arab', 'eng_Latn', 'fra_Latn', 'hin_Deva', 'ind_Latn', 'spa_Latn', 'zho_Hans'],

    # BLOOM/BLOOMZ pretraining (ROOTS) languages, intersected with Belebele's
    # 122 (39 langs). Native scripts only (romanized *_Latn dups dropped);
    # Arabic = MSA arb_Arab only (not the dialectal acm/apc/ars/ary/arz); both
    # Chinese scripts. Excludes 5 ROOTS langs absent from Belebele: Chi Tumbuka,
    # Kikuyu, Akan, Fon, Kirundi. This is the BLOOM analogue of the paper's
    # XLM-R "Seen" source group.
    'bloom_seen': [
        # high-resource / Romance / other
        'eng_Latn', 'fra_Latn', 'spa_Latn', 'por_Latn', 'cat_Latn', 'eus_Latn',
        'ind_Latn', 'vie_Latn', 'arb_Arab', 'zho_Hans', 'zho_Hant',
        # Indic (Indo-Aryan + Dravidian)
        'asm_Beng', 'ben_Beng', 'guj_Gujr', 'hin_Deva', 'kan_Knda', 'mal_Mlym',
        'mar_Deva', 'npi_Deva', 'ory_Orya', 'pan_Guru', 'tam_Taml', 'tel_Telu',
        'urd_Arab',
        # Niger-Congo
        'bam_Latn', 'ibo_Latn', 'kin_Latn', 'lin_Latn', 'lug_Latn', 'nso_Latn',
        'nya_Latn', 'sna_Latn', 'swh_Latn', 'tsn_Latn', 'tso_Latn', 'wol_Latn',
        'xho_Latn', 'yor_Latn', 'zul_Latn',
    ],

    # --- SIB-200 encoder backbones -------------------------------------------
    # The two groups below are the SIB-200 analogues of the paper's "Seen" set
    # (xpe.pdf sec. 4.2: "Seen: The 92 languages that were included in the XLM-R
    # pretraining corpus"). THIS dict is the single source of truth for every
    # group -- read source langs from here, never from src/sib200_meta.py, so a
    # group and the runs it produced can't drift. That module holds the
    # per-language metadata behind these lists (Joshi tier, family, region,
    # which backbone saw what), and tests/test_lang_groups.py pins these literals
    # against it.

    # mDeBERTa-v3 (microsoft/mdeberta-v3-base) is pretrained on CC100 like XLM-R,
    # so this group is taken to BE the published paper's Seen-92 -- which makes
    # mDeBERTa the bridge row: same seen/unseen partition, new code. That is an
    # assumption, not a documented identical language list (the model card says
    # "the 2.5T CC100 data as XLM-R"; neither it nor the DeBERTaV3 paper states a
    # language count, and mDeBERTa uses mT5's vocabulary). See src/sib200_meta.py.
    'mdeberta_seen': [
        'afr_Latn', 'als_Latn', 'amh_Ethi', 'arb_Arab', 'asm_Beng', 'azb_Arab',
        'azj_Latn', 'bel_Cyrl', 'ben_Beng', 'bos_Latn', 'bul_Cyrl', 'cat_Latn',
        'ces_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'ell_Grek', 'eng_Latn',
        'epo_Latn', 'est_Latn', 'eus_Latn', 'fin_Latn', 'fra_Latn', 'gaz_Latn',
        'gla_Latn', 'gle_Latn', 'glg_Latn', 'guj_Gujr', 'hau_Latn', 'heb_Hebr',
        'hin_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ind_Latn', 'isl_Latn',
        'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kan_Knda', 'kat_Geor', 'kaz_Cyrl',
        'khk_Cyrl', 'khm_Khmr', 'kir_Cyrl', 'kmr_Latn', 'kor_Hang', 'lao_Laoo',
        'lit_Latn', 'lvs_Latn', 'mal_Mlym', 'mar_Deva', 'mkd_Cyrl', 'mya_Mymr',
        'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'ory_Orya', 'pan_Guru',
        'pbt_Arab', 'pes_Arab', 'plt_Latn', 'pol_Latn', 'por_Latn', 'ron_Latn',
        'rus_Cyrl', 'san_Deva', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'snd_Arab',
        'som_Latn', 'spa_Latn', 'srp_Cyrl', 'sun_Latn', 'swe_Latn', 'swh_Latn',
        'tam_Taml', 'tel_Telu', 'tha_Thai', 'tur_Latn', 'uig_Arab', 'ukr_Cyrl',
        'urd_Arab', 'uzn_Latn', 'vie_Latn', 'xho_Latn', 'ydd_Hebr', 'zho_Hans',
        'zho_Hant', 'zsm_Latn',
    ],

    # mGTE (Alibaba-NLP/gte-multilingual-mlm-base, Zhang et al. 2024) MLM
    # pretraining data: 75 languages enumerated in Table 7 (App. A.1), all of
    # which land in SIB-200. 76 codes because Chinese is counted as one language
    # there but is two codes here (zho_Hans + zho_Hant), same convention as
    # 'aya_seen'. mGTE's set is nearly a subset of XLM-R's: only ceb/hat/quy/tgl/
    # yor are seen here and unseen for XLM-R, while 21 XLM-R langs are unseen
    # here -- so mgte_unseen strictly extends the paper's Unseen group.
    # Caveat kept in src/sib200_meta.py (MGTE_TOKENS_M): 5 of these are seen only
    # nominally -- quy 0.07M, yor 0.04M, hat 0.03M, jav 0.62M, som 0.82M tokens.
    'mgte_seen': [
        'afr_Latn', 'als_Latn', 'arb_Arab', 'azj_Latn', 'bel_Cyrl', 'ben_Beng',
        'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cym_Latn', 'dan_Latn',
        'deu_Latn', 'ell_Grek', 'eng_Latn', 'est_Latn', 'eus_Latn', 'fin_Latn',
        'fra_Latn', 'glg_Latn', 'guj_Gujr', 'hat_Latn', 'heb_Hebr', 'hin_Deva',
        'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ind_Latn', 'isl_Latn', 'ita_Latn',
        'jav_Latn', 'jpn_Jpan', 'kan_Knda', 'kat_Geor', 'kaz_Cyrl', 'khk_Cyrl',
        'khm_Khmr', 'kir_Cyrl', 'kor_Hang', 'lao_Laoo', 'lit_Latn', 'lvs_Latn',
        'mal_Mlym', 'mar_Deva', 'mkd_Cyrl', 'mya_Mymr', 'nld_Latn', 'nob_Latn',
        'npi_Deva', 'pan_Guru', 'pes_Arab', 'pol_Latn', 'por_Latn', 'quy_Latn',
        'ron_Latn', 'rus_Cyrl', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'som_Latn',
        'spa_Latn', 'srp_Cyrl', 'swe_Latn', 'swh_Latn', 'tam_Taml', 'tel_Telu',
        'tgl_Latn', 'tha_Thai', 'tur_Latn', 'ukr_Cyrl', 'urd_Arab', 'vie_Latn',
        'yor_Latn', 'zho_Hans', 'zho_Hant', 'zsm_Latn',
    ],

    # XLM-R-large (FacebookAI/xlm-roberta-large) -- the PAPER'S OWN BACKBONE, so
    # this group is the published Seen-92 by definition, not by assumption. It is
    # the same literal list as 'mdeberta_seen' (both are the `xlmr` column of
    # src/sib200_meta.py), but the two keys are deliberately separate: the
    # mDeBERTa one rests on "mDeBERTa trains on CC100 like XLM-R", which is an
    # inference from the model card, while this one is the column's actual
    # subject. Do not collapse them into an alias -- if the mDeBERTa assumption
    # is ever revised, only that group should move.
    'xlmr_seen': [
        'afr_Latn', 'als_Latn', 'amh_Ethi', 'arb_Arab', 'asm_Beng', 'azb_Arab',
        'azj_Latn', 'bel_Cyrl', 'ben_Beng', 'bos_Latn', 'bul_Cyrl', 'cat_Latn',
        'ces_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'ell_Grek', 'eng_Latn',
        'epo_Latn', 'est_Latn', 'eus_Latn', 'fin_Latn', 'fra_Latn', 'gaz_Latn',
        'gla_Latn', 'gle_Latn', 'glg_Latn', 'guj_Gujr', 'hau_Latn', 'heb_Hebr',
        'hin_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ind_Latn', 'isl_Latn',
        'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kan_Knda', 'kat_Geor', 'kaz_Cyrl',
        'khk_Cyrl', 'khm_Khmr', 'kir_Cyrl', 'kmr_Latn', 'kor_Hang', 'lao_Laoo',
        'lit_Latn', 'lvs_Latn', 'mal_Mlym', 'mar_Deva', 'mkd_Cyrl', 'mya_Mymr',
        'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'ory_Orya', 'pan_Guru',
        'pbt_Arab', 'pes_Arab', 'plt_Latn', 'pol_Latn', 'por_Latn', 'ron_Latn',
        'rus_Cyrl', 'san_Deva', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'snd_Arab',
        'som_Latn', 'spa_Latn', 'srp_Cyrl', 'sun_Latn', 'swe_Latn', 'swh_Latn',
        'tam_Taml', 'tel_Telu', 'tha_Thai', 'tur_Latn', 'uig_Arab', 'ukr_Cyrl',
        'urd_Arab', 'uzn_Latn', 'vie_Latn', 'xho_Latn', 'ydd_Hebr', 'zho_Hans',
        'zho_Hant', 'zsm_Latn',
    ],
}

# Per-backbone "low-performing" TARGET languages -- the analogue of the paper's
# low-performing group, but defined empirically per decoder backbone (a lang can
# be low-performing for BLOOMZ and not for Aya, so this is NOT one shared list).
# These are TARGET groups, never source groups: they are disjoint from the
# backbone's pretraining-seen set and from joshi5, i.e. low-perf is a strict
# SUBSET of that backbone's unseen targets. Consumed by the result aggregators
# (lang_table marks them `seen = -1`), not by the runner.
LOW_PERF_LANG_GROUPS = {
    'aya': [
        # African (Niger-Congo / Afro-Asiatic / Nilotic)
        'yor_Latn', 'gaz_Latn', 'xho_Latn', 'ibo_Latn', 'nya_Latn', 'tir_Ethi',
        'sot_Latn', 'fuv_Latn', 'wol_Latn', 'zul_Latn', 'amh_Ethi', 'lug_Latn',
        'nso_Latn', 'tsn_Latn', 'luo_Latn', 'lin_Latn', 'som_Latn', 'sna_Latn',
        'ssw_Latn', 'kin_Latn', 'plt_Latn', 'bam_Latn', 'hau_Latn',
        # Asian (non-Latin scripts)
        'bod_Tibt', 'shn_Mymr', 'khk_Cyrl', 'lao_Laoo', 'mya_Mymr', 'khm_Khmr',
        'kac_Latn', 'pbt_Arab', 'sin_Latn', 'tgk_Cyrl', 'tel_Telu', 'mri_Latn',
        # romanized variants of otherwise-seen langs
        'arb_Latn', 'npi_Latn', 'ben_Latn',
    ],

    'bloom': [
        # Asian / non-Latin scripts
        'bod_Tibt', 'sin_Sinh', 'lao_Laoo', 'shn_Mymr', 'kat_Geor', 'khm_Khmr',
        'mya_Mymr', 'kac_Latn', 'hye_Armn', 'khk_Cyrl', 'ckb_Arab', 'kir_Cyrl',
        'pbt_Arab', 'uzn_Latn', 'ilo_Latn', 'mri_Latn',
        # African
        'tir_Ethi', 'amh_Ethi', 'gaz_Latn', 'fuv_Latn', 'luo_Latn', 'som_Latn',
        'hau_Latn', 'ssw_Latn', 'plt_Latn',
        # European
        'mlt_Latn',
        # romanized variants of otherwise-seen langs
        'arb_Latn', 'urd_Latn', 'sin_Latn', 'npi_Latn', 'ben_Latn',
    ],

    # ---------------------------------------------------------------------
    # TEMPORARY, 2026-08-11 -- BORROWED FROM XLM-R, NOT MEASURED PER BACKBONE.
    #
    # Both encoder entries below are copies of the paper's SIB-200
    # Low-Performing group (src/sib200_meta.py::low_perf_langs()), which
    # xpe.pdf sec. 4.2 defines by full fine-tuning of *XLM-R-large* in the
    # original benchmark (accuracy < 60%). It therefore characterises XLM-R,
    # not mDeBERTa and not mGTE. It is registered here as a stand-in so the
    # paper's headline row can be computed from runs we already have; it MUST
    # be replaced by a per-backbone measurement (the planned in-language
    # full-FT sweep) before the write-up leans on it, and every table built
    # from it has to say whose measurement defined the group.
    #
    # NOT one shared list, despite being one list today: 'mdeberta' and 'mgte'
    # are deliberately separate keys so replacing one does not silently move
    # the other. They are ALREADY not identical -- see the mgte note below.
    # ---------------------------------------------------------------------
    'mdeberta': [
        # The full 46. mdeberta_seen == the xlmr column, and the paper's
        # low-perf list is disjoint from it by construction (verified: the
        # intersection is empty), so all 46 survive as unseen targets.
        'ace_Arab', 'aka_Latn', 'arb_Latn', 'ayr_Latn', 'bam_Latn', 'bem_Latn',
        'bjn_Arab', 'bod_Tibt', 'cjk_Latn', 'ckb_Arab', 'dyu_Latn', 'dzo_Tibt',
        'ewe_Latn', 'fon_Latn', 'ibo_Latn', 'kab_Latn', 'kam_Latn', 'kbp_Latn',
        'kik_Latn', 'kin_Latn', 'kmb_Latn', 'knc_Arab', 'lua_Latn', 'lug_Latn',
        'min_Arab', 'mni_Beng', 'mos_Latn', 'mri_Latn', 'nqo_Nkoo', 'nso_Latn',
        'nus_Latn', 'run_Latn', 'sat_Olck', 'shn_Mymr', 'smo_Latn', 'sna_Latn',
        'sot_Latn', 'ssw_Latn', 'taq_Latn', 'taq_Tfng', 'tgk_Cyrl', 'tsn_Latn',
        'tso_Latn', 'tzm_Tfng', 'umb_Latn', 'yor_Latn',
    ],

    'mgte': [
        # The same list MINUS yor_Latn, which is in mGTE's MLM pretraining set
        # (mgte_seen, 76 codes) while XLM-R-large scored it below 60%. Low-perf
        # is a STRICT SUBSET OF UNSEEN by definition -- `seen == -1` refines
        # `seen == 0` -- and add_seen() raises on a lang that is in both, so a
        # verbatim copy of the 46 would crash unify for the enarzho and joshi5
        # source groups (for src=mgte_seen, yor_Latn is a source and so was
        # never a target anyway).
        #
        # CONSEQUENCE: the mDeBERTa low-perf row covers 46 languages and the
        # mGTE one covers 45. The two rows are NOT over the same language set,
        # so they are not directly comparable across backbones. Say so wherever
        # they appear side by side.
        'ace_Arab', 'aka_Latn', 'arb_Latn', 'ayr_Latn', 'bam_Latn', 'bem_Latn',
        'bjn_Arab', 'bod_Tibt', 'cjk_Latn', 'ckb_Arab', 'dyu_Latn', 'dzo_Tibt',
        'ewe_Latn', 'fon_Latn', 'ibo_Latn', 'kab_Latn', 'kam_Latn', 'kbp_Latn',
        'kik_Latn', 'kin_Latn', 'kmb_Latn', 'knc_Arab', 'lua_Latn', 'lug_Latn',
        'min_Arab', 'mni_Beng', 'mos_Latn', 'mri_Latn', 'nqo_Nkoo', 'nso_Latn',
        'nus_Latn', 'run_Latn', 'sat_Olck', 'shn_Mymr', 'smo_Latn', 'sna_Latn',
        'sot_Latn', 'ssw_Latn', 'taq_Latn', 'taq_Tfng', 'tgk_Cyrl', 'tsn_Latn',
        'tso_Latn', 'tzm_Tfng', 'umb_Latn',   # yor_Latn removed: mGTE-seen
    ],

    # ---------------------------------------------------------------------
    # XLM-R-large -- the ONE entry here that is NOT borrowed.
    #
    # xpe.pdf sec. 4.2 defines Low-Performing as the SIB-200 languages on which
    # full fine-tuning of *XLM-R-large* scores below 60% in the original
    # benchmark. That measurement is OF this backbone, so for 'xlmr' the list is
    # the definition rather than a stand-in. Tables built from this row need no
    # "whose measurement defined the group" footnote -- unlike the 'mdeberta'
    # and 'mgte' rows above, which still owe a per-backbone full-FT sweep.
    #
    # The full 46 survive: the list was derived from the same `xlmr` column that
    # defines 'xlmr_seen', so it is disjoint from it by construction and
    # add_seen()'s seen/low-perf clash check cannot fire.
    # ---------------------------------------------------------------------
    'xlmr': [
        'ace_Arab', 'aka_Latn', 'arb_Latn', 'ayr_Latn', 'bam_Latn', 'bem_Latn',
        'bjn_Arab', 'bod_Tibt', 'cjk_Latn', 'ckb_Arab', 'dyu_Latn', 'dzo_Tibt',
        'ewe_Latn', 'fon_Latn', 'ibo_Latn', 'kab_Latn', 'kam_Latn', 'kbp_Latn',
        'kik_Latn', 'kin_Latn', 'kmb_Latn', 'knc_Arab', 'lua_Latn', 'lug_Latn',
        'min_Arab', 'mni_Beng', 'mos_Latn', 'mri_Latn', 'nqo_Nkoo', 'nso_Latn',
        'nus_Latn', 'run_Latn', 'sat_Olck', 'shn_Mymr', 'smo_Latn', 'sna_Latn',
        'sot_Latn', 'ssw_Latn', 'taq_Latn', 'taq_Tfng', 'tgk_Cyrl', 'tsn_Latn',
        'tso_Latn', 'tzm_Tfng', 'umb_Latn', 'yor_Latn',
    ],
}
