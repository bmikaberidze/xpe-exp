from micm_nlp import utils

"""
Plot PCA/TSNE of vocabulary embeddings versus soft prompt embeddings.

Usage:
    python -m scripts.evals.plot --config xlmr/finetune/peft/sib200_plot.xpe

"""
import os

import wandb
import warnings
import pprint
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from micm_nlp.models.model import MODEL
import matplotlib.patheffects as pe
from scripts.evals.utils import extract_grouped_hs_embeddings
from src.config_utils import trained_model_paths, grouped_trained_models

import matplotlib
import itertools

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
import numpy as np
import torch
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from openTSNE import TSNE as OpenTSNE # !pip install openTSNE

eval_stor_path = "artefacts/evals"
os.makedirs(os.path.dirname(f"{eval_stor_path}/temp/"), exist_ok=True)
os.makedirs(os.path.dirname(f"{eval_stor_path}/plots/"), exist_ok=True)
os.makedirs(os.path.dirname(f"{eval_stor_path}/plots/script_labels/"), exist_ok=True)

# Set environment variables
# export OMP_NUM_THREADS=8
utils.env["OMP_NUM_THREADS"] = "128"
utils.env["MKL_NUM_THREADS"] = "128"
# print(utils.env["MKL_NUM_THREADS"])
# exit()

FIGSIZE = (11, 7)
VOCAB_KEEP_ONLY = None
VOCAB_KEEP_ONLY = 0 # 5000 # 1000, 5000
SAVE_VOCAB_LABELS_SEPARATELY = False
VOCAB_DOT_SCALE = 0.15
SOFT_DOT_SCALE = 10
SKIP_MIXED_DOTS = True

# Languages and scripts
LANGUAGES = 'seen' # number or group of languages (seen/unseen)
SHOW_LATN_LANGS = True
GROUPED_VOCAB_CACHE = f"{eval_stor_path}/temp/grouped_vocab_cache_{LANGUAGES}"
if SHOW_LATN_LANGS:
    GROUPED_VOCAB_CACHE += "_latn"
GROUPED_VOCAB_CACHE += ".pkl"

PLACEHOLDER_STR = "placeholder"
PROJECTED_CACHE = f"{eval_stor_path}/temp/projected_embedds_{PLACEHOLDER_STR}.npz"

VOCAB_MARKER = "o"
SOFT_MARKER = "*"
COLORS = {
    "SPT": "black",
    "XPE": "goldenrod",
    # 
    "SPT_J5": "green",
    "SPT_SEEN": "blue",
    "XPE_J5": "orange",
    "XPE_SEEN": "red",
}
DECOMPOSITION_TYPE = "open-tsne"
DECOMPOSITION_DICT = {
    "pca": PCA(n_components=2),
    "tsne": TSNE(
        n_components=2, 
        perplexity=30, 
        init="pca", 
        learning_rate="auto"
    ),
    "open-tsne": OpenTSNE(
        n_components=2, 
        perplexity=30, 
        initialization="pca", 
        learning_rate="auto", 
        n_jobs=int(utils.env["OMP_NUM_THREADS"]),
        random_state=42
    ),
}

SOFT_PROMPT_HS_PATH = f"{MODEL.stor_path}/xlmr/FacebookAI|xlm-roberta-large/prompt_hidden_states/prompt_hidden_states_5_13.pt"

spt_low_latn = [
# 'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@0', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@1', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@12', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@13', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@23', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@24', 
]
spt_low_arab = [
# 'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@0', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@1', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@12', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@13', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@23', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@ckb_Arab@24', 
]
spt_low_cyrl = [
# 'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@0', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@1', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@12', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@13', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@23', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@tgk_Cyrl@24', 
]
spt_seen_arab = [
# 'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@0', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@1', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@12', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@13', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@23', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@pes_Arab@24', 
]
spt_seen_latn = [
# 'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@0', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@1', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@12', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@13', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@23', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@tur_Latn@24', 
]
spt_seen_cyrl = [
# 'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@0', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@1', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@12', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@13', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@23', 
'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@seen@ukr_Cyrl@24', 
]
xpe_low_latn = [
# 'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@0', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@1', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@12', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@13', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@23', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@24', 
]
xpe_low_arab = [
# 'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@0', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@1', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@12', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@13', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@23', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@ckb_Arab@24', 
]
xpe_low_cyrl = [
# 'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@0', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@1', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@12', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@13', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@23', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@tgk_Cyrl@24', 
]
xpe_seen_arab = [
# 'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@0', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@1', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@12', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@13', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@23', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@pes_Arab@24', 
]
xpe_seen_latn = [
# 'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@0', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@1', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@12', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@13', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@23', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@tur_Latn@24', 
]
xpe_seen_cyrl = [
# 'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@0', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@1', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@12', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@13', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@23', 
'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@seen@ukr_Cyrl@24'
]

spt_low = spt_low_latn + spt_low_arab + spt_low_cyrl
spt_seen = spt_seen_arab + spt_seen_latn + spt_seen_cyrl
xpe_low = xpe_low_latn + xpe_low_arab + xpe_low_cyrl
xpe_seen = xpe_seen_arab + xpe_seen_latn + xpe_seen_cyrl

CHECK_PROJECTED_CACHE = False
SOFTP_KEEP_ONLY_GROUPS = None # spt_low # spt_low + spt_seen + xpe_low + xpe_seen
# SOFTP_KEEP_ONLY_GROUPS = ["SPT", "XPE"] \
# SOFTP_KEEP_ONLY_GROUPS = [] \
# + spt_low_latn \
# + spt_low_arab \
# + spt_low_cyrl \
# + spt_seen_arab \
# + spt_seen_latn \
# + spt_seen_cyrl \
# + xpe_low_latn \
# + xpe_low_arab \
# + xpe_low_cyrl \
# + xpe_seen_arab \
# + xpe_seen_latn \
# + xpe_seen_cyrl 
# SOFTP_KEEP_ONLY_GROUPS = [
#     'mikaberidze/xlmr-large-sib200-peft-xpe-joshi5@low-perf@bem_Latn@0', 
#     'mikaberidze/xlmr-large-sib200-peft-spt-joshi5@low-perf@bem_Latn@0', 
# ]

# SOFTP_KEEP_ONLY_ENDS = ["@1", "@2"]
SOFTP_KEEP_ONLY_MODEL = None
SOFTP_KEEP_ONLY_GROUP = None
SOFTP_KEEP_ONLY_LAYERS = None
# SOFTP_KEEP_ONLY_ENDS = ["@23", "@24"]

FILE_NAME = None
# print(FILE_NAME)
# exit()

def get_peft_model_path_by_id(id):
    if id in trained_model_paths:
        return trained_model_paths[id]
    else:
        raise ValueError(f"PEFT model path not found for id: {id}")

id_by_groups = {
    "SPT": grouped_trained_models["SPT_SEEN"] + grouped_trained_models["SPT_J5"],
    "XPE": grouped_trained_models["XPE_SEEN"] + grouped_trained_models["XPE_J5"]
}

def get_color_by_group(group):
    if "Cyrl" in group:
        return "purple"
    elif "Arab" in group:
        return "orange"
    elif "Latn" in group:
        return "green"
    elif "Beng" in group:
        return "black"
    elif "Mymr" in group:
        return "red"
    else:
        return "black"

def _get_color_by_group(group):
    if group in COLORS:
        return COLORS[group]
    else:
        color = "black"
        is_xpe = "xpe" in group

        if group.endswith("@0"):
            color = "yellow" if is_xpe else "lime"

        elif group.endswith("@1"):
            color = "orange" if is_xpe else "green"

        elif '@12' in group or '@13' in group:
            color = "red" if is_xpe else "blue"

        elif '@23' in group or '@24' in group:
            color = "pink" if is_xpe else "purple"

        return color

def project_and_plot(
    vocab_embedds: np.ndarray,
    vocab_labels: np.ndarray,
    softp_embedds: np.ndarray,
    softp_labels: np.ndarray,
    output_path: str):
    """
    Plots:
      - Vocab tokens grouped by script (colored)
      - Mixed tokens in gray
      - Soft prompt groups in COLORS[group]
      - Soft prompts if ungrouped → black
    """

    # ---------------------------------------------
    # 0. Apply VOCAB_KEEP_ONLY 
    # ---------------------------------------------
    if VOCAB_KEEP_ONLY is not None and vocab_embedds.shape[0] > VOCAB_KEEP_ONLY:
        idx = np.random.choice(vocab_embedds.shape[0], VOCAB_KEEP_ONLY, replace=False)
        vocab_embedds = vocab_embedds[idx]
        vocab_labels = vocab_labels[idx]

    # ---------------------------------------------
    # 1. Apply SOFTP_KEEP_ONLY
    # ---------------------------------------------
    # print(SOFTP_KEEP_ONLY_MODEL, SOFTP_KEEP_ONLY_GROUP, SOFTP_KEEP_ONLY_LAYERS)
    # return
    if SOFTP_KEEP_ONLY_GROUPS is not None:
        idx = np.array([i for i, label in enumerate(softp_labels) if label in SOFTP_KEEP_ONLY_GROUPS])
        softp_embedds = softp_embedds[idx]
        softp_labels = softp_labels[idx]

    elif SOFTP_KEEP_ONLY_MODEL and SOFTP_KEEP_ONLY_GROUP and SOFTP_KEEP_ONLY_LAYERS:
        idx = np.array([i for i, label in enumerate(softp_labels) 
            if SOFTP_KEEP_ONLY_MODEL in label and 
            SOFTP_KEEP_ONLY_GROUP in label and 
            any(layer in label for layer in SOFTP_KEEP_ONLY_LAYERS)])

        softp_embedds = softp_embedds[idx]
        softp_labels = softp_labels[idx]

    print(
        f"[INFO] \nvocab_embedds={vocab_embedds.shape}, softp_embedds={softp_embedds.shape} \n"
        f"vocab_labels={len(vocab_labels)}, softp_labels={len(softp_labels)}"
    )
    # exit()

    # ---------------------------------------------
    # 2. Stack for PCA/TSNE
    # ---------------------------------------------
    print(f"[INFO] len(vocab_embedds)={len(vocab_embedds)}, len(softp_embedds)={len(softp_embedds)}")
    if len(vocab_embedds) != 0 and len(softp_embedds) != 0:
        stacked = np.vstack([vocab_embedds, softp_embedds])
    elif len(vocab_embedds) != 0:
        stacked = np.vstack(vocab_embedds)
    elif len(softp_embedds) != 0:
        stacked = np.vstack(softp_embedds)
    else:
        raise ValueError("No embeddings to stack")

    projected_cache_path = PROJECTED_CACHE.replace(PLACEHOLDER_STR, f"{len(vocab_labels)}_{len(softp_labels)}")
    if CHECK_PROJECTED_CACHE and os.path.exists(projected_cache_path):
        saved = np.load(projected_cache_path)
        projected_embedds = saved["projected_embedds"]
        print(f"[INFO] Loaded projected embeddings from: {projected_cache_path}")
    else:
        if "tsne" in DECOMPOSITION_TYPE:
            print("[TSNE] PCA")
            X50 = PCA(n_components=50).fit_transform(stacked)
            print("[TSNE] TSNE")
            if "open-tsne" in DECOMPOSITION_TYPE:
                projected_embedds = DECOMPOSITION_DICT["open-tsne"].fit(X50)
            else:
                projected_embedds = DECOMPOSITION_DICT["tsne"].fit_transform(X50)
        else:
            print("[PCA] PCA only")
            projected_embedds = DECOMPOSITION_DICT["pca"].fit_transform(stacked)
        np.savez(projected_cache_path, projected_embedds=projected_embedds)
        print(f"[INFO] Saved projected embeddings to: {projected_cache_path}")

    proj_vocab_embedds = projected_embedds[:len(vocab_embedds)]
    proj_softp_embedds = projected_embedds[len(vocab_embedds):]

    plot(proj_vocab_embedds, proj_softp_embedds, vocab_labels, softp_labels, output_path)

def plot(proj_vocab_embedds: np.ndarray, proj_softp_embedds: np.ndarray, vocab_labels: np.ndarray, softp_labels: np.ndarray, output_path: str):
    '''
    Plots:
      - Vocab tokens grouped by label (colored)
      - Mixed tokens in gray
      - Soft prompt groups in COLORS[label]
      - Soft prompts if ungrouped → black
    '''
    print(f"[INFO] Plotting... vocab={len(vocab_labels)}, softp={len(softp_labels)}")
    # ---------------------------------------------
    # 1. Plotting 2D
    # ---------------------------------------------
    plt.figure(figsize=FIGSIZE)

    # colors for label groups
    vocab_label_set = sorted(set(vocab_labels.tolist()))
    color_map = build_script_color_map(vocab_label_set)

    # ---- vocab tokens (scripts + mixed)
    for label in vocab_label_set:
        mask = vocab_labels == label
        plt.scatter(
            proj_vocab_embedds[mask, 0],
            proj_vocab_embedds[mask, 1],
            s=VOCAB_DOT_SCALE, 
            alpha=0.5, 
            color=color_map[label], 
            label=label
        )

    # =============================================
    # SAVE LABELS
    # =============================================
    HALO_LW = 2.5 
    if not SAVE_VOCAB_LABELS_SEPARATELY:
        labels = []
        for label in vocab_label_set:
            if label in ["Mixed"]:
                continue

            mask = vocab_labels == label
            if mask.sum() == 0:
                continue

            pts = proj_vocab_embedds[mask]
            cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])

            label_font_size = 6 if SHOW_LATN_LANGS and "Latn" in label else 12

            txt = plt.text(
                cx, cy, label,
                fontsize=label_font_size,
                fontweight="bold",
                color=color_map[label],
                ha="center",
                va="center",
                zorder=10,
            )

            txt.set_path_effects([
                pe.Stroke(linewidth=HALO_LW, foreground="white"),
                pe.Normal()
            ])

            labels.append({
                "text": txt,
                "x": cx,
                "y": cy,
                "anchor": np.array([cx, cy])  # attract back to cluster
            })

            relax_labels(labels)

    else:

        script_labels_dir = f"{eval_stor_path}/plots/{FILE_NAME}/script_labels/"

        for label in vocab_label_set:

            fig, ax = plt.subplots(figsize=(2, 1), dpi=300)
            fig.patch.set_alpha(0)     # transparent figure background
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            txt = ax.text(
                0.5, 0.5, label,
                ha="center", va="center",
                fontsize=48,
                fontweight="bold",
                color=color_map[label],
            )

            # thin white outline around letters (no bbox!)
            txt.set_path_effects([
                pe.Stroke(linewidth=HALO_LW, foreground="white"),
                pe.Normal()
            ])

            # Save tightly cropped, transparent
            svg_path = os.path.join(script_labels_dir, f"{label}.svg")
            png_path = os.path.join(script_labels_dir, f"{label}.png")

            fig.savefig(svg_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
            fig.savefig(png_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

    # ---- SOFT PROMPT GROUPS
    softp_label_set = sorted(set(softp_labels.tolist()))
    for label in softp_label_set:
        mask = softp_labels == label
        color = get_color_by_group(label)
        plt.scatter(
            proj_softp_embedds[mask, 0],
            proj_softp_embedds[mask, 1],
            s=SOFT_DOT_SCALE,
            alpha=0.9,
            color=color,
            label=label,
            marker=SOFT_MARKER,
            edgecolors="none"
        )

    # Legend outside + bigger markers
    handles, labels = plt.gca().get_legend_handles_labels()

    softp_handles = []
    softp_labels = []

    for h, l in zip(handles, labels):
        # if l in COLORS or l == "Mixed":
        if l in COLORS:
            softp_handles.append(h)
            softp_labels.append(l)

    plt.legend(
        softp_handles,
        softp_labels,
        markerscale=5,
        scatterpoints=1,
        # bbox_to_anchor=(1.05, 1),
        loc="lower right"
    )

    # Standardize figure size and export vector + raster outputs
    plt.gcf().set_size_inches(*FIGSIZE)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    base, _ = os.path.splitext(output_path)
    svg_path = f"{base}.svg"
    png_path = f"{base}.png"

    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {os.path.abspath(svg_path)} and {os.path.abspath(png_path)}")

def relax_labels(labels, iters=200, repel=0.02, attract=0.01,
                 step=0.05, max_radius=0.25):

    for _ in range(iters):
        for i, a in enumerate(labels):
            force = np.zeros(2)
            pa = np.array([a["x"], a["y"]])

            # repel from other labels
            for j, b in enumerate(labels):
                if i == j:
                    continue
                pb = np.array([b["x"], b["y"]])
                d = pa - pb
                dist = np.linalg.norm(d)
                if dist < 0.12:
                    force += (d / (dist + 1e-6)) * repel

            # attraction to cluster anchor
            force += (a["anchor"] - pa) * attract

            # apply damping
            delta = np.clip(force, -step, step)
            a["x"] += delta[0]
            a["y"] += delta[1]

            # hard constraint: stay near anchor
            offset = np.array([a["x"], a["y"]]) - a["anchor"]
            dist = np.linalg.norm(offset)
            if dist > max_radius:
                a["x"], a["y"] = a["anchor"] + offset / dist * max_radius

    for a in labels:
        a["text"].set_position((a["x"], a["y"]))

def build_script_color_map(unique_labels):
    """
    Assigns each script a stable, visually high-contrast color.
    'Mixed' is always gray.
    """
    palette = (
        list(plt.cm.tab20.colors) +
        list(plt.cm.Set1.colors) +
        list(plt.cm.Set2.colors) +
        list(plt.cm.Dark2.colors)
    )

    palette_iter = itertools.cycle(palette)
    color_map = {}

    for label in unique_labels:
        if label == "Mixed":
            color_map[label] = "gray"
        else:
            color_map[label] = next(palette_iter)

    return color_map

def collect_softp_embedds_by_groups(config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all PEFT models listed in id_by_groups using the provided config,
    and collect:
      - one shared vocab_embedds (from the first model),
      - stacked soft embeddings for all models,
      - group_labels (string per soft token row).
    """
    from micm_nlp.models.model import MODEL
    from micm_nlp.datasets.dataset import DATASET
    from micm_nlp.tokenizers.tokenizer import load as load_tokenizer

    tokenizer = load_tokenizer(config)
    dataset = DATASET(config, tokenizer)

    calc_vocab_embedds = True
    vocab_embedds = None
    softp_embedds_list = []
    softp_labels_list = []

    for group_name, uuid_list in id_by_groups.items():
        for uid in uuid_list:
            encoder_path = get_peft_model_path_by_id(uid)
            # Update config to point to this adapter
            config.task.peft.encoder_init_state_dict_path = encoder_path
            config.task.peft.encoder_ratio = 0 if group_name.startswith("SPT") else 1
            model = MODEL(config, tokenizer, dataset)  # NOTE: same config/tokenizer/dataset reused

            ve, se = extract_embeddings(model.get(), calc_vocab_embedds)
            if calc_vocab_embedds:
                vocab_embedds = ve  # keep vocab only from the first model
                calc_vocab_embedds = False

            softp_embedds_list.append(se)
            softp_labels_list.extend([group_name] * se.shape[0])

            # free model ASAP
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if vocab_embedds is None:
        raise RuntimeError("No vocab embeddings collected – check id_by_groups / paths.")

    softp_embedds = np.vstack(softp_embedds_list)
    softp_labels = np.array(softp_labels_list, dtype=object)
    print("[STEP] collect_softp_embedds_by_groups: Completed collection.", flush=True)
    print(f"  - softp_embedds shape: {softp_embedds.shape}", flush=True)
    print(f"  - softp_labels shape: {softp_labels.shape}", flush=True)
    return softp_embedds, softp_labels, vocab_embedds

def extract_embeddings(peft_model, calc_vocab_embedds: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return base vocab embeddings and soft prompt embeddings from a PEFT model."""
    base_model = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    if calc_vocab_embedds:
        vocab_embedds = base_model.get_input_embeddings().weight.detach().cpu().numpy()
    else:
        vocab_embedds = None

    with torch.no_grad():
        prompt = peft_model.get_prompt(batch_size=1).detach().cpu().squeeze(0)
    softp_embedds = prompt.numpy()

    return vocab_embedds, softp_embedds

def extract_grouped_vocab_embedds(
    vocab_embedds: np.ndarray,
    base_model_name="FacebookAI/xlm-roberta-large",
    sib200_name="Davlan/sib200",
    max_texts_per_lang=200,
    verbose=True):
    """
    Returns:
        grouped_vocab_token_ids: dict {script_name → set(token_ids)}
    """

    # -----------------------------------------------------
    # 0. Check for CACHE
    # -----------------------------------------------------
    if os.path.exists(GROUPED_VOCAB_CACHE):
        if verbose: print(f"[CACHE] Loading cached vocab groups from {GROUPED_VOCAB_CACHE}")
        with open(GROUPED_VOCAB_CACHE, "rb") as f:
            grouped_vocab_token_ids = pickle.load(f)

    else:

        if verbose: print("[CACHE] No cache found. Running extraction...")

        # -----------------------------------------------------
        # 1. Load tokenizer
        # -----------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # -----------------------------------------------------
        # 2. Read dataset configs
        # -----------------------------------------------------
        configs = get_dataset_config_names(sib200_name)
        configs = [c for c in configs if not c.endswith("Nkoo.zip")]
        if LANGUAGES: 
            if isinstance(LANGUAGES, int):
                configs = configs[:LANGUAGES]
            elif LANGUAGES == "seen":
                from src.sib200_meta import xlmr_seen_sib200_ds_names
                configs = xlmr_seen_sib200_ds_names

        if verbose: print(f"[INFO] Found {len(configs)} SIB-200 language configs.")

        # Group languages by script suffix
        script_to_configs = defaultdict(list)

        for cfg in configs:
            if "_" in cfg:
                _, script = cfg.split("_", 1)
                if SHOW_LATN_LANGS and script == "Latn":
                    script = cfg
            else:
                script = "UNK"
            script_to_configs[script].append(cfg)
        
        if verbose: print(f"[INFO] Scripts: {list(script_to_configs.keys())}")

        # -----------------------------------------------------
        # 3. Tokenize datasets & gather token IDs per script
        # -----------------------------------------------------
        grouped_vocab_token_ids = defaultdict(set)

        for script, cfg_list in script_to_configs.items():
            if verbose: print(f"\n[INFO] Processing script '{script}' ({len(cfg_list)} languages)")

            for cfg in cfg_list:
                try:
                    dataset = load_dataset(sib200_name, cfg, split="train")
                except Exception as e:
                    print(f"[WARN] Failed to load {cfg}: {e}")
                    continue

                for i, example in enumerate(dataset):
                    text = (
                        example.get("text")
                        or example.get("sentence")
                        or example.get("inputs")
                    )

                    if not text:
                        continue

                    encoded = tokenizer(
                        text,
                        truncation=True,
                        max_length=128,
                        add_special_tokens=True,
                    )

                    ids = encoded["input_ids"]

                    if isinstance(ids[0], list):
                        for seq in ids:
                            grouped_vocab_token_ids[script].update(seq)
                    else:
                        grouped_vocab_token_ids[script].update(ids)

                    if i + 1 >= max_texts_per_lang:
                        break

            if verbose: print(f"  → {len(grouped_vocab_token_ids[script])} unique tokens")

        # -----------------------------------------------------
        # 4. Compute mixed tokens (appear in >1 script)
        # -----------------------------------------------------
        token_to_scripts = defaultdict(set)
        for script, tokens in grouped_vocab_token_ids.items():
            for tid in tokens:
                token_to_scripts[tid].add(script)

        mixed_tokens = {tid for tid, scrs in token_to_scripts.items() if len(scrs) > 1}

        if verbose: print(f"[INFO] Mixed tokens: {len(mixed_tokens)}")

        # Remove mixed tokens from individual script sets
        for script in grouped_vocab_token_ids:
            grouped_vocab_token_ids[script] -= mixed_tokens

        # Add mixed tokens to the grouped_vocab_token_ids dictionary
        grouped_vocab_token_ids["Mixed"] = mixed_tokens

        if verbose:
            for script, toks in grouped_vocab_token_ids.items():
                print(f"Final unique tokens for '{script}': {len(toks)}")

        # -----------------------------------------------------
        # 5. SAVE CACHE
        # -----------------------------------------------------
        with open(GROUPED_VOCAB_CACHE, "wb") as f:
            pickle.dump(grouped_vocab_token_ids, f)

        if verbose: print(f"[CACHE] Saved cache to {GROUPED_VOCAB_CACHE}")


    # Token IDs → vocab embeddings
    grouped_vocab_embedds = {
        group_name: vocab_embedds[np.array(list(token_ids), dtype=np.int64)]
        for group_name, token_ids in grouped_vocab_token_ids.items()
        if len(token_ids) > 0
    }

    # Merge ALL vocab embeddings (scripts + mixed) and labels
    grouped_vocab_embedds_list = []
    grouped_vocab_labels_list = []
    for script, emb in grouped_vocab_embedds.items():
        if script == "Mixed" and SKIP_MIXED_DOTS:
            continue
        grouped_vocab_embedds_list.append(emb)
        grouped_vocab_labels_list.extend([script] * len(emb))

    grouped_vocab_embedds = np.vstack(grouped_vocab_embedds_list)
    grouped_vocab_labels = np.array(grouped_vocab_labels_list, dtype=object)

    return grouped_vocab_embedds, grouped_vocab_labels

def run(config, output_path=f"{eval_stor_path}/plots/{FILE_NAME}.png"):
    print("[STEP] run_colorful_langs: Starting...")

    # Get soft prompt embeddings + soft prompt group labels + base vocab
    grouped_softp_embedds, grouped_softp_labels, vocab_embedds = collect_softp_embedds_by_groups(config)

    # Extract grouped soft prompt hidden state embeddings
    grouped_softp_hs_embedds, grouped_softp_hs_labels = extract_grouped_softp_hs_embeddings(SOFT_PROMPT_HS_PATH)
    grouped_softp_embedds = np.concatenate([grouped_softp_embedds, grouped_softp_hs_embedds], axis=0)
    grouped_softp_labels = np.concatenate([grouped_softp_labels, grouped_softp_hs_labels], axis=0)
    # utils.p(sorted(list(set(grouped_softp_labels.tolist()))))
    # exit()

    # Vocab embeddings grouped by script or language
    grouped_vocab_embedds, grouped_vocab_labels = extract_grouped_vocab_embedds(vocab_embedds)

    # Plot with script colors + mixed + soft prompt groups
    project_and_plot(
        vocab_embedds=grouped_vocab_embedds,
        vocab_labels=grouped_vocab_labels,
        softp_embedds=grouped_softp_embedds,
        softp_labels=grouped_softp_labels,
        output_path=output_path,
    )

def run_hs(config):
    global SOFTP_KEEP_ONLY_MODEL, SOFTP_KEEP_ONLY_GROUP, SOFTP_KEEP_ONLY_LAYERS, FILE_NAME
    print("[STEP] run hidden states: Starting...")

    # Extract grouped soft prompt hidden state embeddings
    grouped_softp_hs_embedds, grouped_softp_hs_labels = extract_grouped_softp_hs_embeddings(SOFT_PROMPT_HS_PATH)

    for model in ["xpe", "spt"]:
        for group in ["low-perf", "seen"]:
            for layers in [["@1","@2"],["@12","@13"],["@23","@24"]]:

                SOFTP_KEEP_ONLY_MODEL = model 
                SOFTP_KEEP_ONLY_GROUP = group
                SOFTP_KEEP_ONLY_LAYERS = layers

                FILE_NAME = f"{SOFTP_KEEP_ONLY_MODEL}-j5-{SOFTP_KEEP_ONLY_GROUP}-hs-" + "-".join(SOFTP_KEEP_ONLY_LAYERS)
                print("FILE_NAME: ", FILE_NAME)

                output_path=f"{eval_stor_path}/plots/hs/{FILE_NAME}.png"

                project_and_plot(
                    vocab_embedds=np.empty((0, 0)), # pass empty array to skip vocab embeddings
                    vocab_labels=np.empty((0,)),
                    softp_embedds=grouped_softp_hs_embedds,
                    softp_labels=grouped_softp_hs_labels,
                    output_path=output_path
                )
                # exit()

def main():
    from micm_nlp.enums import ConfigTypeSE
    from micm_nlp.config import CONFIG

    config_name = utils.parse_script_args()
    config = CONFIG.load(config_name, ConfigTypeSE.LANGUAGE_MODEL)

    # run(config)
    run_hs(config)

    wandb.finish()


if __name__ == "__main__":
    main()
