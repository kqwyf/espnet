#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

VOX_PATH=/home/chenda/ironwolf/database/voxceleb2
LRS2=~/ironwolf/database/lrs2/mvlrs_v1/

stage=3
stop_stage=3

dir=$(pwd)

cd ../../; 
. ./path.sh

cd ${dir}

cmd=run.pl
nj=16

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for dset in dev test; do
        mkdir -p ./data/${dset}
        find ${VOX_PATH}/${dset} | grep '\.mp4' > ./data/${dset}/flist
        awk -F '[/.]' '{print $(NF-3)"-"$(NF-2)"-"$(NF-1), $N}' data/${dset}/flist > data/${dset}/video.scp
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    dset=pretrain
    mkdir -p ./data/LRS2/${dset}
    awk  -v lrs2=${LRS2} -F '[/ ]' '{print $1"_"$2, lrs2"/pretrain/"$1"/"$2".mp4"}' ${LRS2}/${dset}.txt | sort > data/LRS2/${dset}/video.scp
    dset=val
    mkdir -p ./data/LRS2/${dset}
    awk  -v lrs2=${LRS2} -F '[/ ]' '{print $1"_"$2, lrs2"/main/"$1"/"$2".mp4"}' ${LRS2}/${dset}.txt | sort > data/LRS2/${dset}/video.scp
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    for dataset in pretrain val; do
        
        log_dir=data/LRS2/${dataset}/split_${nj}
        split_scps=""
        mkdir -p ${log_dir}
        for n in $(seq $nj); do
            split_scps="$split_scps ${log_dir}/split.$n.scp"
        done
        ./utils/split_scp.pl data/LRS2/${dataset}/video.scp $split_scps || exit 1
        
        $cmd JOB=1:$nj ${log_dir}/format_video.JOB.log python ./py_tools/format_video.py \
        ${log_dir}/split.JOB.scp \
        scp,ark:${log_dir}/video.JOB.scp,${log_dir}/video.JOB.ark ${log_dir}/audio.JOB|| exit 1

        cat ${log_dir}/audio.*/wav.scp | sort | uniq > data/LRS2/${dataset}/wav.scp
        cat ${log_dir}/video.*.scp | sort | uniq > data/LRS2/${dataset}/video_npy.scp
        
    done
fi


