#!/bin/bash
# Generate one XLT group config: one grid x one source group = one output dir.
#
# The matrix (4 methods x 3 folds x 10 seeds) is identical across source groups, so
# the files differ only in the source group they name -- which is why they are
# generated rather than hand-maintained. See docs/micm-nlp-0.4-migration.md.
#
#   scripts/gen_xlt_group_config.sh 14a_bebe_grid_full_aya enarzho aya \
#       > config/groups/14a_bebe_grid_full_aya.enarzho.yml
#
# Args: <grid> <source_group> <llm> [units_dir_relative_to_groups]
set -euo pipefail

GRID=${1:?grid name, e.g. 14a_bebe_grid_full_aya}
SRC=${2:?source group, e.g. enarzho}
LLM=${3:?backbone tag, e.g. aya}
UNITS=${4:-../units}

cat <<EOF
# ${GRID}, source group ${SRC^^} (${LLM}).
#
# Generated: scripts/gen_xlt_group_config.sh ${GRID} ${SRC} ${LLM}
# One group config per source group: one experiment, one output directory
# (artefacts/runs/groups/${GRID}.${SRC}/), so a group's runs can never mix two
# source sets. Every entry key beyond the reserved ones is stamped on each result
# row, so method / fold / source_group / llm are columns of the aggregate table.
#
# Axes: method (xpe, spt, d30, d70) x fold (0,1,2) x seed (10..19) = 120 runs.
# The DUAL arms differ from each other only by peft.encoder_ratio.
#
# Dispatch (Aya needs >80 GB VRAM -> pin the big-VRAM partitions; --mem is host RAM):
#   sbatch --array=0-119%10 --mem=80G \\
#     --partition=B200,H200,H200-PCI,H100-PCI \\
#     runtime/clusters/pegasus/shell/run.sh --site-packages \\
#       "python -m micm_nlp run-group \\
#          --group-config config/groups/${GRID}.${SRC}.yml \\
#          --runner scripts.xlt_runner:run \\
#          --root-path /fscratch/bmikaberidze/xpe-exp"

configs:
  xpe:  ${UNITS}/tune.xpe.lm.${LLM}.ds.bebe.yml
  spt:  ${UNITS}/tune.spt.lm.${LLM}.ds.bebe.yml
  dual: ${UNITS}/tune.dual.lm.${LLM}.ds.bebe.yml
  test: ${UNITS}/test.lm.${LLM}.ds.bebe.fold.yml

runs:
EOF

for m in xpe spt d30 d70; do
    case $m in
        xpe) cfg=xpe;  ov='' ;;
        spt) cfg=spt;  ov='' ;;
        d30) cfg=dual; ov=', overrides: {peft.encoder_ratio: 0.3}' ;;
        d70) cfg=dual; ov=', overrides: {peft.encoder_ratio: 0.7}' ;;
    esac
    printf '\n  # ===== %s =====\n' "$m"
    for f in 0 1 2; do
        for s in $(seq 10 19); do
            printf '  - {config: %-5s name: %s, seed: %d, fold: %d, method: %s, source_group: %s, llm: %s%s, separate_test: {config: test}}\n' \
                "${cfg}," "${m}_f${f}_s${s}" "$s" "$f" "$m" "$SRC" "$LLM" "$ov"
        done
    done
done
