source ./llm_common_eval/batch_eval.sh
SCRIPT=eval_mistral/mistral_on_SuperGLUE_all_4bit.py

# Dataset validation size (ordered):
# cb        56
# copa      100
# wsc       104
# rte       277
# wic       638
# boolq     3.27K
# multirc   4.85K
# record    10K

export CUDA_VISIBLE_DEVICES=0
detached_experiment exp1 \
    python $SCRIPT --dataset cb
waitfor_experiments exp1

export CUDA_VISIBLE_DEVICES=0
detached_experiment exp2 \
    python $SCRIPT --dataset copa
export CUDA_VISIBLE_DEVICES=1
detached_experiment exp3 \
    python $SCRIPT --dataset wsc
export CUDA_VISIBLE_DEVICES=2
detached_experiment exp4 \
    python $SCRIPT --dataset rte
export CUDA_VISIBLE_DEVICES=3
detached_experiment exp5 \
    python $SCRIPT --dataset wic
export CUDA_VISIBLE_DEVICES=4
detached_experiment exp6 \
    python $SCRIPT --dataset boolq
export CUDA_VISIBLE_DEVICES=5
detached_experiment exp7 \
    python $SCRIPT --dataset multirc
waitfor_experiments exp2 exp3 exp4 exp5 exp6 exp7

#detached_experiment exp8 \
#    python $SCRIPT --dataset record
#waitfor_experiments exp8
