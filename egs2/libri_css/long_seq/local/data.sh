#!/bin/bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Chenda Li)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


LIBRICSS_SIM=/mnt/lustre/sjtu/home/cdl54/share/jsalt/simu_meeting_9k



train_set='train'
dev_set='dev'
test_set='test'


for folder in ${test_set} ${dev_set} ${train_set}; do
  mkdir -p data/${folder}
done

  
find ${LIBRICSS_SIM} | grep test_ | grep mix | while read line; do
  echo `basename ${line} .mix.wav` ${line}
done | sort  > data/${test_set}/wav.scp

find ${LIBRICSS_SIM} | grep test_  | grep ref1 | while read line; do
  echo `basename ${line} .ref1.wav` ${line}
done | sort  > data/${test_set}/spk1.scp

find ${LIBRICSS_SIM} | grep test_  | grep ref2 | while read line; do
  echo `basename ${line} .ref2.wav` ${line}
done | sort  > data/${test_set}/spk2.scp


find ${LIBRICSS_SIM} | grep dev_  | grep mix | while read line; do
  echo `basename ${line} .mix.wav` ${line}
done | sort   > data/${dev_set}/wav.scp

find ${LIBRICSS_SIM} | grep dev_  | grep ref1 | while read line; do
  echo `basename ${line} .ref1.wav` ${line}
done | sort  > data/${dev_set}/spk1.scp

find ${LIBRICSS_SIM} | grep dev_  | grep ref2 | while read line; do
  echo `basename ${line} .ref2.wav` ${line}
done | sort  > data/${dev_set}/spk2.scp


find ${LIBRICSS_SIM} | grep train_ | grep mix | while read line; do
  echo `basename ${line} .mix.wav` ${line}
done | sort   > data/${train_set}/wav.scp

find ${LIBRICSS_SIM} | grep train_  | grep ref1 | while read line; do
  echo `basename ${line} .ref1.wav` ${line}
done | sort  > data/${train_set}/spk1.scp

find ${LIBRICSS_SIM} | grep train_ | grep ref2 | while read line; do
  echo `basename ${line} .ref2.wav` ${line}
done | sort  > data/${train_set}/spk2.scp

for folder in ${test_set} ${dev_set} ${train_set}; do
  cat data/${folder}/wav.scp | awk '{print $1, $1}' > data/${folder}/utt2spk
  cat data/${folder}/wav.scp | awk '{print $1, $1}' > data/${folder}/spk2utt
done
