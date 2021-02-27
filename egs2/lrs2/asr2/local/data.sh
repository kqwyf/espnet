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

. ./db.sh
. ./path.sh
. ./cmd.sh

cmd=run.pl
nj=32
stage=1
stop_stage=6

min_or_max=max # min, max, or fix
pretrain_model=lrw # lrw or unsupervised or none
pretrain_model_path=./local/unsupervised/exp/models/3.pt
sample_rate=8k

log "$0 $*"
. utils/parse_options.sh


if [ ! -e "${LRS2}" ]; then
    log "Fill the value of 'LRS2' of db.sh"
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for dataset in train val test; do
        mkdir -p data/${dataset}
        awk  -v lrs2=${LRS2} -F '[/ ]' '{print $1"_"$2, lrs2"/main/"$1"/"$2".mp4"}' ${LRS2}/${dataset}.txt | sort > data/${dataset}/video.scp
        awk '{print $1, "ffmpeg -i " $2 " -ar 16000 -ac 1  -f wav pipe:1 |" }' data/${dataset}/video.scp > data/${dataset}/wav.scp
        awk '{print $2}' data/${dataset}/video.scp | sed -e 's/.mp4/.txt/g' | while read line 
        do 
            grep 'Text:' $line | sed -e 's/Text:  //g'
        done > data/${dataset}/text_tmp
        paste  <(awk '{print $1}' data/${dataset}/wav.scp)  data/${dataset}/text_tmp >  data/${dataset}/text
        rm data/${dataset}/text_tmp
        awk '{print $1, $1}' data/${dataset}/wav.scp > data/${dataset}/utt2spk
        awk '{print $1, $1}' data/${dataset}/wav.scp > data/${dataset}/spk2utt

    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ${pretrain_model} = 'lrw' ]; then
    # Download the pretrained model to extract visual features.
    if [ ! -f ./local/feature_extract/finetuneGRU_19.pt ]; then
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12ww9Vp8q3g-PdNGKvgEW8dPmFlJdhko0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12ww9Vp8q3g-PdNGKvgEW8dPmFlJdhko0" -O ./local/feature_extract/finetuneGRU_19.pt.tgz && rm -rf /tmp/cookies.txt
        tar xzvf ./local/feature_extract/finetuneGRU_19.pt.tgz -C ./local/feature_extract/
    fi
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ ${pretrain_model} != 'none' ]; then
    for dataset in  train val test; do
        echo "extracting visual feature for [${dataset}]"
        _nj=8
        _num_folds=1 # number of gpus will be _nj / _num_folds
        log_dir=data/${dataset}/split_${_nj}
        split_scps=""
        mkdir -p ${log_dir}
        for n in $(seq ${_nj}); do
            split_scps="$split_scps ${log_dir}/video.$n.scp"
        done
        ./utils/split_scp.pl data/${dataset}/video.scp $split_scps || exit 1
        
        if [ ${pretrain_model} = 'lrw' ]; then
            $cmd JOB=1:${_nj} ${log_dir}/extract_visual_feature.JOB.log python ./local/feature_extract/extract_visual_feature.py \
            --job-index JOB --jobs-per-gpu ${_num_folds} \
            ${log_dir}/video.JOB.scp \
            scp,ark:${log_dir}/vfeature.JOB.scp,${log_dir}/vfeature.JOB.ark || exit 1
        elif [ ${pretrain_model} = 'unsupervised' ]; then
            $cmd JOB=1:${_nj} ${log_dir}/extract_visual_feature.JOB.log python ./local/unsupervised/extract_v_from_pretrain.py ${pretrain_model_path} \
            --job-index JOB --jobs-per-gpu ${_num_folds} \
            ${log_dir}/video.JOB.scp \
            scp,ark:${log_dir}/vfeature.JOB.scp,${log_dir}/vfeature.JOB.ark || exit 1
        fi
        
        for n in $(seq ${_nj}); do
            cat ${log_dir}/vfeature.${n}.scp
        done > data/${dataset}/vfeature.scp
    done
fi



if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    for dataset in train val test; do
        echo "create mixture for ${dataset}"

        mkdir -p local/mixture_list
        ./local/generate_mixture_list.py < data/${dataset}/wav.scp > local/mixture_list/${dataset}
        
        log_dir=data/${dataset}/split_${nj}_${min_or_max}
        split_scps=""
        mkdir -p ${log_dir}
        for n in $(seq $nj); do
            split_scps="$split_scps ${log_dir}/pairlist.$n"
        done
        ./utils/split_scp.pl local/mixture_list/${dataset} $split_scps || exit 1
        
        
        $cmd JOB=1:$nj ${log_dir}/create_mixture.JOB.log ./local/create_mixture.py --min_or_max ${min_or_max} data/${dataset}/wav.scp data/${dataset}/text ${log_dir}/pairlist.JOB ${log_dir}/mixture_wavs.JOB/wav ${log_dir}/mixture_wavs.JOB || exit 1
        
        mkdir -p data/${dataset}_${min_or_max}_2mix
        cat ${log_dir}/mixture_wavs.*/wav.scp | sort > data/${dataset}_${min_or_max}_2mix/wav.scp
        cat ${log_dir}/mixture_wavs.*/spk1.scp | sort > data/${dataset}_${min_or_max}_2mix/spk1.scp
        cat ${log_dir}/mixture_wavs.*/spk2.scp | sort > data/${dataset}_${min_or_max}_2mix/spk2.scp
        cat ${log_dir}/mixture_wavs.*/text_spk1 | sort > data/${dataset}_${min_or_max}_2mix/text_spk1
        cat ${log_dir}/mixture_wavs.*/text_spk2 | sort > data/${dataset}_${min_or_max}_2mix/text_spk2
        
        awk '{print $1,$1}' data/${dataset}_${min_or_max}_2mix/wav.scp > data/${dataset}_${min_or_max}_2mix/utt2spk
        awk '{print $1,$1}' data/${dataset}_${min_or_max}_2mix/wav.scp > data/${dataset}_${min_or_max}_2mix/spk2utt
        
    done
    
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    
    if [ ${pretrain_model} != 'none' ];  then
        for dataset in train val test; do
            echo "pick parallel visual features for ${dataset}"
            
            log_dir=data/${dataset}/split_${nj}_${min_or_max}
            split_scps=""
            mkdir -p ${log_dir}
            for n in $(seq $nj); do
                split_scps="$split_scps ${log_dir}/pairlist.$n"
            done
            ./utils/split_scp.pl local/mixture_list/${dataset} $split_scps || exit 1
            
            
            $cmd JOB=1:$nj ${log_dir}/pick_vfeatrure.JOB.log ./local/pick_vfeature.py --min_or_max ${min_or_max} data/${dataset}/vfeature.scp ${log_dir}/pairlist.JOB ${log_dir}/parallel_vfearure.JOB || exit 1
            
            mkdir -p data/${dataset}_${min_or_max}_2mix
            cat ${log_dir}/parallel_vfearure.*/v1.scp | sort > data/${dataset}_${min_or_max}_2mix/v1.scp
            cat ${log_dir}/parallel_vfearure.*/v2.scp | sort > data/${dataset}_${min_or_max}_2mix/v2.scp
            
        done
    else
        for dataset in train val test; do
            echo "pick parallel videos for ${dataset}"
            
            log_dir=data/${dataset}/split_${nj}_${min_or_max}
            split_scps=""
            mkdir -p ${log_dir}
            for n in $(seq $nj); do
                split_scps="$split_scps ${log_dir}/pairlist.$n"
            done
            ./utils/split_scp.pl local/mixture_list/${dataset} $split_scps || exit 1
            
            
            $cmd JOB=1:$nj ${log_dir}/pick_vfeatrure.JOB.log ./local/pick_videos.py --min_or_max ${min_or_max} data/${dataset}/video.scp ${log_dir}/pairlist.JOB ${log_dir}/pick_video.JOB || exit 1
            
            mkdir -p data/${dataset}_${min_or_max}_2mix
            cat ${log_dir}/pick_video.*/video1.scp | sort > data/${dataset}_${min_or_max}_2mix/video1.scp
            cat ${log_dir}/pick_video.*/video2.scp | sort > data/${dataset}_${min_or_max}_2mix/video2.scp
        done
    fi
fi

#if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
#        for dataset in train val test; do
#            echo "pick parallel contextual features for ${dataset}"
#            
#            log_dir=data/${dataset}/split_${nj}_${min_or_max}
#            split_scps=""
#            mkdir -p ${log_dir}
#            for n in $(seq $nj); do
#                split_scps="$split_scps ${log_dir}/pairlist.$n"
#            done
#            ./utils/split_scp.pl local/mixture_list/${dataset} $split_scps || exit 1
#            
#            
#            $cmd JOB=1:$nj ${log_dir}/pick_o_ctx.JOB.log ./local/pick_vfeature.py --min_or_max ${min_or_max} --output_name o_ctx data/${dataset}/o_ctx_main.scp ${log_dir}/pairlist.JOB ${log_dir}/parallel_o_ctx.JOB || exit 1
#            
#            mkdir -p data/${dataset}_${min_or_max}_2mix
#            cat ${log_dir}/parallel_o_ctx.*/o_ctx1.scp | sort > data/${dataset}_${min_or_max}_2mix/o_ctx1.scp
#            cat ${log_dir}/parallel_o_ctx.*/o_ctx2.scp | sort > data/${dataset}_${min_or_max}_2mix/o_ctx2.scp
#            
#        done
#
#fi

