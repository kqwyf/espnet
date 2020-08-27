#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=max # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=16k


train_set="train"
valid_set="dev"
test_sets="test "

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --lang en \
    --ngpu 4 \
    --enh_config ./conf/tuning/train_enh_PSM_debug.yaml \
    "$@"
