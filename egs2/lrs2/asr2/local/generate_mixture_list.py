#!/usr/bin/env python3

import random
import argparse

random.seed(1234)

num_folds = 2

utt_ids = []
while True:
    try:
        utt_ids.append(input().split()[0])
    except EOFError:
        break

utt_ids = [utt_id.split('_') for utt_id in utt_ids] # (dataset, spkr_id, utt_id)

for i, (dataset, spkr_id, utt_id) in enumerate(utt_ids):
    for _ in range(num_folds):
        target_index = random.randrange(len(utt_ids))
        while utt_ids[target_index][1] == spkr_id:
            target_index = random.randrange(len(utt_ids))
        snr = random.random() * 10 - 5 # (-5, 5)
        print('_'.join(utt_ids[i]) + '_' + '_'.join(utt_ids[target_index]) + '_' + ('%.3f' % snr) + '_' + ('%.3f' % (-snr)))

