. ./path.sh

enh_exp=./exp/tasnet_8k
inference_model=91epoch.pth
python -m espnet2.bin.enh_inference \
                    --ngpu "0" \
                    --fs "8k" \
                    --data_path_and_name_and_type "./demo/mix.scp,speech_mix,sound" \
                    --key_file "./demo/mix.scp"\
                    --enh_train_config "${enh_exp}"/config.yaml \
                    --enh_model_file "${enh_exp}"/"${inference_model}" \
                    --normalize_output_wav true \
                    --output_dir "${enh_exp}"/separated \
