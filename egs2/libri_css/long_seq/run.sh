#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k


train_set="train_3w"
valid_set="dev_3w"
test_sets="librscss_seg "
# test_sets="test_ov60_clean "

./enh.sh \
    --max_wav_duration 500 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --lang en \
    --inference_model valid.loss.ave.pth \
    --ngpu 1 \
    --gpu_inference true \
    --inference_nj 1 \
    --enh_config ./conf/tuning/base_psm_big.yaml \
    "$@"
