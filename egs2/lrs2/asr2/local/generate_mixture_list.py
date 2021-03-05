#!/usr/bin/env python3

import sys
import random

random.seed(1234)
num_folds = 2

def main(argv):
    utt2num_samples_file = argv[1]
    length_rate = float(argv[2])

    utt2num_samples = {}
    with open(utt2num_samples_file, 'r', encoding='utf-8') as f:
        for line in f:
            utt, num_samples = line.split()
            utt2num_samples[utt] = int(num_samples)

    utt_labels = []
    while True:
        try:
            utt_labels.append(input().split()[0])
        except EOFError:
            break
    num_utts = len(utt_labels)

    # sort utts by length
    utt_labels.sort(key=lambda x: utt2num_samples[x], reverse=True)

    mixture_set = set()
    mixture_list = []

    lower_bound = 0
    upper_bound = 1
    for upper_bound in range(num_utts):
        if utt2num_samples[utt_labels[upper_bound]] < utt2num_samples[utt_labels[0]] * (1 - length_rate):
            break
    
    for utt_label in utt_labels:
        _, spkr_id, _ = utt_label.split('_') # (dataset, spkr_id, utt_id)

        while utt2num_samples[utt_labels[lower_bound]] > utt2num_samples[utt_label] * (1 + length_rate):
            lower_bound += 1

        while upper_bound < num_utts and utt2num_samples[utt_labels[upper_bound]] > utt2num_samples[utt_label] * (1 - length_rate):
            upper_bound += 1

        if upper_bound - lower_bound < num_folds + 1: # too few selections available, skip it
            continue

        for _ in range(num_folds):
            target_index = random.randrange(lower_bound, upper_bound)
            mix_label = utt_label + '_' + utt_labels[target_index]
            mix_label2 = utt_labels[target_index] + '_' + utt_label
            while utt_labels[target_index].split('_')[1] == spkr_id or mix_label in mixture_set or mix_label2 in mixture_set: # speaker collision or generation collision
                target_index = random.randrange(lower_bound, upper_bound)
                mix_label = utt_label + '_' + utt_labels[target_index]
                mix_label2 = utt_labels[target_index] + '_' + utt_label
            mixture_set.add(mix_label)
            snr = random.random() * 10 - 5 # (-5, 5)
            snr_label = ("%.3f" % snr) + "_" + ("%.3f" % (-snr))
            mixture_list.append((mix_label, snr_label))

    for mix_label, snr_label in mixture_list:
        print(mix_label + '_' + snr_label)

if __name__ == "__main__":
    main(sys.argv)

