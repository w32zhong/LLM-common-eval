source ./llm_common_eval/batch_eval.sh

detached_experiment exp1 \
    python ./eval_phi2/phi2_on_CNN_Dailymail.py \
        --log_endpoint my_cloudflare_r2 --devices 0 --skip_until 0
waitfor_experiments exp1
